import re
import cv2
import pygame
from .constants import WIDTH, HEIGHT, SQUARE_SIZE, RED, WHITE, BLACK
from .camera.ximea_camera import XimeaCamera
from .camera.board_detection_copy import BoardDetection
from .checkers.game import Game
from .minimax.algorithm import WHITE, minimax
import mediapipe as mp
import time
import numpy as np
import threading
from enum import Enum, auto

from checkers_msgs.msg import Board, Piece, Move, HandDetected, RobotMove

import rclpy
from rclpy.node import Node


class Difficulty(Enum):
    """AI difficulty levels"""
    EASY = 1      # Depth 1 - Very simple, makes mistakes
    MEDIUM = 3    # Depth 3 - Balanced gameplay
    HARD = 5      # Depth 5 - Very challenging


class GameState(Enum):
    """Game state machine states"""
    WAITING_FOR_START = auto()
    PLAYER_TURN = auto()
    DETECTING_PLAYER_MOVE = auto()
    VALIDATING_PLAYER_MOVE = auto()
    AI_THINKING = auto()
    ROBOT_MOVING = auto()
    WAITING_FOR_ROBOT = auto()


class CheckersNode(Node):
    def __init__(self):
        super().__init__('checkers_game')
        print("\n" + "="*60)
        print("INITIALIZING CHECKERS NODE")
        print("="*60)
        
        pygame.init()
        self.WIN = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Checkers")
        self.clock = pygame.time.Clock()
        self.FPS = 60
        
        print("Creating camera connection...")
        self.ximeaCamera = XimeaCamera()
        print("✓ Camera connected")
        
        print("\nInitializing board detection...")
        print("(This will open corner selection window)")
        self.boardDetection = BoardDetection(self.ximeaCamera)
        print("✓ Board detection ready")
        
        self.game = Game(self.WIN, self.boardDetection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State machine
        self.game_state = GameState.WAITING_FOR_START
        self.game_started = False
        
        # Robot synchronization - using locks for thread safety
        self.robot_lock = threading.Lock()
        self.isRobotMoveDone = True
        self.waiting_for_robot = False
        self.pending_player_move = False  # Flag if player moved during robot turn
        
        # Detection state with threading
        self.detection_lock = threading.Lock()
        self.latest_board_detection = None
        self.latest_camera_image = None
        self.detection_thread_running = True
        
        # Movement detection
        self.movement_detected = False
        self.movement_stable_time = None
        self.STABILITY_THRESHOLD = 0.8  # Reduced from 1.0 for faster response
        
        # AI Difficulty setting
        self.difficulty = Difficulty.MEDIUM  # Default difficulty
        self.ai_depth = self.difficulty.value
        
        # Hand detection cooldown to prevent false positives
        self.last_hand_detected_time = 0
        self.HAND_COOLDOWN = 0.5  # 500ms cooldown after hand leaves

        # Publishers and subscribers - using absolute topic names
        self.board_publisher = self.create_publisher(Board, '/board_topic', 10)
        self.move_publisher = self.create_publisher(Move, '/move_topic', 10)
        self.hand_detected_publisher = self.create_publisher(HandDetected, '/hand_detected', 10)
        self.robot_move_subscription = self.create_subscription(
            RobotMove, '/robot_move_topic', self.robot_move_callback, 10)

        self.oldCameraImage = self.ximeaCamera.get_camera_image()
        
        print("\n" + "="*60)
        print("SYSTEM READY - Starting main game loop")
        print("="*60)
        print("\n→ Set AI difficulty using the trackbar in calibration window")
        print("→ Press SPACE to confirm calibration, then 'S' to START")
        print("→ Press SPACE during gameplay to force move validation")
        print(f"\n→ Default difficulty: {self.difficulty.name} (depth={self.ai_depth})\n")
        
        # Start background detection thread
        # NOTE: Detection thread only gets camera images, NOT OpenCV windows
        # OpenCV windows must be handled in main thread
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Create a ROS2 timer to handle pygame events every 1/FPS seconds
        self.timer = self.create_timer(1/self.FPS, self.update_pygame)

    def _detection_loop(self):
        """Background thread for continuous camera image capture ONLY.
        NOTE: DO NOT call any cv2.imshow() or boardDetection.get_board() here!
        OpenCV GUI operations must be in the main thread."""
        while self.detection_thread_running:
            try:
                # Get camera image ONLY - no OpenCV window operations!
                camera_image = self.ximeaCamera.get_camera_image()
                
                with self.detection_lock:
                    self.latest_camera_image = camera_image
                
                # Small sleep to prevent CPU overload
                time.sleep(0.02)  # ~50Hz capture rate
                
            except Exception as e:
                self.get_logger().error(f"Detection thread error: {e}")
                time.sleep(0.1)
    
    def get_latest_detection(self):
        """Thread-safe getter for latest camera image"""
        with self.detection_lock:
            return self.latest_camera_image, self.latest_board_detection

    def robot_move_callback(self, msg):
        """Callback when robot finishes its move"""
        with self.robot_lock:
            self.isRobotMoveDone = msg.robot_move_done
            
        if msg.robot_move_done:
            print("\n" + "="*60)
            print("✓ ROBOT FINISHED - Your turn!")
            print("="*60)
            
            # Check if player already made a move during robot turn
            if self.pending_player_move:
                print("→ Player move was detected during robot turn, validating now...")
                self.pending_player_move = False
                self._force_validate_move()
            else:
                print("→ Make your move on the board")
                print("→ Remove your hand and wait for detection\n")
            
            self.game_state = GameState.PLAYER_TURN
            self.waiting_for_robot = False
            self.movement_detected = False
            self.movement_stable_time = None
    
    def set_difficulty(self, difficulty: Difficulty):
        """Set AI difficulty level"""
        self.difficulty = difficulty
        self.ai_depth = difficulty.value
        print(f"\n★ Difficulty set to: {difficulty.name} (search depth={self.ai_depth})")
    
    def _force_validate_move(self):
        """Force immediate move validation"""
        cameraImage, boardDetected = self.get_latest_detection()
        if boardDetected is not None:
            self._validate_and_process_move(boardDetected)

    def update_pygame(self):
        """Main game loop controlled by ROS2 timer"""
        # Handle pygame quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.detection_thread_running = False
                self.destroy_node()
                pygame.quit()
                return

        self.clock.tick(self.FPS)
        
        # Get latest camera image from background thread
        cameraImage, _ = self.get_latest_detection()
        
        if cameraImage is None:
            # Fallback: get image directly if thread hasn't provided one yet
            cameraImage = self.ximeaCamera.get_camera_image()
            if cameraImage is None:
                return
        
        # Run board detection in MAIN THREAD (required for OpenCV windows)
        # ALWAYS run detection - even before game starts, for calibration
        boardDetected = self.boardDetection.get_board(cameraImage, self.game)
        with self.detection_lock:
            self.latest_board_detection = boardDetected
            
        # Get board coordinates and check for hand
        board_coordinates = self.get_board_coordinates()
        isHandDetected = self.detect_hand(cameraImage, board_coordinates)

        if isHandDetected:
            self.last_hand_detected_time = time.time()
            self.publish_hand_detected_state()

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Difficulty selection (1, 2, 3)
        if key == ord('1'):
            self.set_difficulty(Difficulty.EASY)
        elif key == ord('2'):
            self.set_difficulty(Difficulty.MEDIUM)
        elif key == ord('3'):
            self.set_difficulty(Difficulty.HARD)
        
        # Press 'S' to start the game
        if key == ord('s') or key == ord('S'):
            if not self.game_started:
                self._start_game()
        
        # Press SPACE to force move validation
        if key == 32:
            print("\n[SPACE pressed] Forcing move validation...")
            self._force_validate_move()
        
        # Only process game logic if game has started
        if not self.game_started:
            return
        
        # State machine processing
        self._process_game_state(cameraImage, boardDetected, isHandDetected)
    
    def _start_game(self):
        """Initialize and start the game"""
        # Read difficulty from calibration if set
        if hasattr(self.boardDetection, 'selected_difficulty'):
            calibrated_depth = self.boardDetection.selected_difficulty
            if calibrated_depth <= 1:
                self.difficulty = Difficulty.EASY
            elif calibrated_depth <= 3:
                self.difficulty = Difficulty.MEDIUM
            else:
                self.difficulty = Difficulty.HARD
            self.ai_depth = calibrated_depth
            print(f"\n★ Using calibrated AI difficulty: {self.difficulty.name} (depth={self.ai_depth})")
        
        self.game_started = True
        self.game_state = GameState.PLAYER_TURN
        print("\n" + "="*60)
        print("GAME STARTED - Make your first move!")
        print("="*60)
        print(f"\nDifficulty: {self.difficulty.name} (AI depth={self.ai_depth})")
        print("\nYOU ARE WHITE (bottom 3 rows)")
        print("You can move WHITE pieces diagonally forward")
        print("\nValid starting positions for WHITE:")
        print("  Row 5: positions 40, 41, 42, 43")
        print("  Row 6: positions 48, 49, 50, 51")
        print("  Row 7: positions 56, 57, 58, 59")
        print("\nEach piece can move diagonally forward to an empty square")
        print("Example: Piece at position 40 → can move to position 32 or 33")
        print("\n→ Move a piece, then remove your hand completely")
        print("→ Wait for board to stabilize (0.8 seconds)")
        print("="*60 + "\n")
    
    def _process_game_state(self, cameraImage, boardDetected, isHandDetected):
        """Process game based on current state"""
        
        # Check for winner
        if self.game.winner() is not None:
            print("\n" + "="*60)
            print(f"GAME OVER - Winner: {self.game.winner()}")
            print("="*60 + "\n")
            return
        
        # Get robot status safely
        with self.robot_lock:
            robot_done = self.isRobotMoveDone
        
        # Handle different states
        if self.game_state == GameState.WAITING_FOR_ROBOT:
            # Waiting for robot to complete its move
            if not robot_done:
                # Check if player is trying to move during robot turn
                if not isHandDetected and self._detect_board_change(cameraImage):
                    print("⚠ Movement detected during robot turn - will validate after robot finishes")
                    self.pending_player_move = True
            return
        
        if self.game_state == GameState.AI_THINKING:
            # AI is calculating - this state transitions automatically
            return
        
        if self.game_state == GameState.PLAYER_TURN:
            if boardDetected is None:
                return
                
            # Wait for hand cooldown
            if time.time() - self.last_hand_detected_time < self.HAND_COOLDOWN:
                return
            
            # Detect if there's movement on the board
            isChange = self._detect_board_change(cameraImage)
            
            if isChange and not isHandDetected:
                # Movement detected without hand - start stability timer
                if not self.movement_detected:
                    self.movement_detected = True
                    self.movement_stable_time = time.time()
                    print("\n→ Movement detected on board, waiting for stabilization...")
                    self._print_detection_summary(boardDetected)
                else:
                    # Reset timer if movement continues
                    self.movement_stable_time = time.time()
            elif not isChange and self.movement_detected:
                # No change detected - check if stable long enough
                if self.movement_stable_time and \
                   time.time() - self.movement_stable_time > self.STABILITY_THRESHOLD:
                    print("✓ Board stable - validating move...")
                    self._validate_and_process_move(boardDetected)
                    self.movement_detected = False
                    self.movement_stable_time = None
            elif isHandDetected:
                # Hand detected - reset movement detection
                if self.movement_detected:
                    print("  Hand detected - resetting stability timer")
                self.movement_detected = False
                self.movement_stable_time = None
        
        # Update the old camera image for next comparison
        self.oldCameraImage = cameraImage
    
    def _detect_board_change(self, newCameraImage):
        """Detect if there's significant change on the board"""
        if self.oldCameraImage is None:
            return False
            
        diff = cv2.absdiff(newCameraImage, self.oldCameraImage)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        if len(thresh.shape) == 3:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        
        change_percentage = np.sum(thresh == 255) / thresh.size * 100
        significant_change_threshold = 2.0
        
        # Display the difference image
        message = f"Change: {change_percentage:.1f}%" if change_percentage > significant_change_threshold else f"Stable: {change_percentage:.1f}%"
        cv2.putText(thresh, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Difference', thresh)
        
        return change_percentage > significant_change_threshold
    
    def _print_detection_summary(self, boardDetected):
        """Print current detection summary"""
        if boardDetected is None:
            return
        black_count = np.count_nonzero(boardDetected == 2)
        white_count = np.count_nonzero(boardDetected == 1)
        print(f"  Current detection: Black={black_count}, White={white_count}")
    
    def _validate_and_process_move(self, boardDetected):
        """Validate player's move and process AI response"""
        print("\n→ Validating move against game rules...")
        is_valid = self.game.wasItValidMove(boardDetected)
        
        if is_valid:
            print("✓ Valid move detected!")
            self.game.update(boardDetected)
            self.publish_board_state()
            
            # Check if it's AI's turn (BLACK)
            if self.game.board.isBoardCreated and self.game.turn == BLACK:
                self._execute_ai_turn()
            else:
                print("\n→ Your move complete! Waiting for next move...\n")
                self.game_state = GameState.PLAYER_TURN
        else:
            self._handle_invalid_move(boardDetected)
    
    def _execute_ai_turn(self):
        """Execute AI's turn with current difficulty"""
        self.game_state = GameState.AI_THINKING
        print("\n" + "-"*60)
        print(f"AI's turn - Calculating best move (depth={self.ai_depth}, {self.difficulty.name})...")
        
        try:
            # Run minimax with current difficulty depth
            value, new_board, new_move, new_piece, removed = minimax(
                self.game.get_board(), 
                self.ai_depth, 
                BLACK, 
                self
            )
            
            if new_move is None or new_piece is None:
                print("⚠ AI couldn't find a valid move!")
                self.game_state = GameState.PLAYER_TURN
                return
            
            print(f"✓ AI decided to move from ({new_piece.row},{new_piece.col}) to {new_move}")
            
            self.aiMove = new_move
            self.aiPiece = new_piece
            self.aiPiece.color = BLACK
            self.game.update(new_board.board)
            
            # Publish move to robot
            self.publish_move_state(new_move, new_piece, removed)
            
            # Update state to waiting for robot
            with self.robot_lock:
                self.isRobotMoveDone = False
            self.game_state = GameState.WAITING_FOR_ROBOT
            self.waiting_for_robot = True
            
            print("Waiting for robot to execute move...")
            print("-"*60 + "\n")
            
        except Exception as e:
            self.get_logger().error(f"AI error: {e}")
            print(f"⚠ AI error: {e}")
            self.game_state = GameState.PLAYER_TURN
    
    def _handle_invalid_move(self, boardDetected):
        """Handle invalid move detection"""
        black_count = np.count_nonzero(boardDetected == 2)
        white_count = np.count_nonzero(boardDetected == 1)
        
        print("\n" + "="*60)
        print("✗ INVALID MOVE - Move rejected")
        print("="*60)
        print(f"  Detected pieces: Black={black_count}, White={white_count}")
        print(f"  Current turn: {'WHITE (you)' if self.game.turn == WHITE else 'BLACK (AI)'}")
        
        self._analyze_board_changes(boardDetected)
        
        if self.game.turn == WHITE:
            self._print_valid_moves()
        
        print("="*60 + "\n")
        
        # Stay in player turn state
        self.game_state = GameState.PLAYER_TURN
    
    def _print_valid_moves(self):
        """Print all valid moves for the current player"""
        print("\n  YOUR TURN - Move a WHITE piece:")
        try:
            white_pieces = self.game.board.get_all_pieces_by_color(WHITE)
            
            if white_pieces and len(white_pieces) > 0:
                movable_pieces = []
                for piece in white_pieces:
                    valid_moves = self.game.board.get_valid_moves(piece)
                    if valid_moves and len(valid_moves) > 0:
                        movable_pieces.append((piece, valid_moves))
                
                if len(movable_pieces) > 0:
                    print(f"  You have {len(movable_pieces)} WHITE pieces that can move:")
                    for piece, moves in movable_pieces:
                        row, col = piece.row, piece.col
                        position = row * 8 + col
                        move_positions = [move[0] * 8 + move[1] for move in moves.keys()]
                        print(f"    • Piece at position {position} (row {row}, col {col}) → can move to: {move_positions}")
                else:
                    print("  ⚠ No valid moves available for WHITE pieces!")
            else:
                print("  ⚠ No WHITE pieces found on board!")
        except Exception as e:
            print(f"  ⚠ Error getting valid moves: {e}")

    def findDifference(self, newCameraImage, oldCameraImage):
        image2 = oldCameraImage
        image1 = newCameraImage

        diff = cv2.absdiff(image1, image2)

        # Threshold the diff image so we get only significant changes
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Optionally apply some dilation to make the differences more visible
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Convert to grayscale if the images are color images
        if len(thresh.shape) == 3:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # Calculate the percentage of change
        change_percentage = np.sum(thresh == 255) / thresh.size * 100

        # Increase threshold for significant change to reduce false positives
        significant_change_threshold = 2.0

        isChange = False
        if change_percentage > significant_change_threshold:
            message = f"Change: {change_percentage:.1f}%"
            isChange = True
        else:
            message = f"Stable: {change_percentage:.1f}%"
            isChange = False

        # Display the result with text
        cv2.putText(thresh, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Difference', thresh)

        return isChange

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def get_board_coordinates(self):
        board_coordinates = []
        for row in range(len(self.boardDetection.gameBoardFieldsContours)):
            for col in range(len(self.boardDetection.gameBoardFieldsContours[row])):
                boundary = self.boardDetection.gameBoardFieldsContours[row][col]
                # boundary is already [x, y, w, h], no need for cv2.boundingRect
                x, y, w, h = boundary
                centerX = x + w / 2
                centerY = y + h / 2
                board_coordinates.append([centerX, centerY])
        return board_coordinates    

    def detect_hand(self, cameraImage, board_coordinates):
        bgr_frame = cv2.cvtColor(cameraImage, cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(rgb_frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
                min_x, min_y, max_x, max_y = self.get_hand_bounding_box(landmarks, rgb_frame)
                cv2.rectangle(rgb_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                cv2.imshow("Hand Tracking", rgb_frame)
                if self.do_rectangles_overlap((min_x, min_y, max_x - min_x, max_y - min_y), board_coordinates):
                    return True
                else:
                    return False
        cv2.imshow("Hand Tracking", rgb_frame)
        return False

    @staticmethod
    def do_rectangles_overlap(rect1, rect2, threshold=50):
        if len(rect1) != 4 or len(rect2) != 4:
            return False
        
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return (x1 - threshold < x2 + w2) and (x1 + w1 + threshold > x2) and (y1 - threshold < y2 + h2) and (y1 + h1 + threshold > y2)

    @staticmethod
    def get_hand_bounding_box(landmarks, frame):
        min_x = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
        max_x = int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
        min_y = int(min(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
        max_y = int(max(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
        return min_x, min_y, max_x, max_y


    def color_to_string(self, color):
        if color == WHITE:
            return "white"
        elif color == BLACK:
            return "red"
        
    def publish_board_state(self):    
        board_msg = Board()
        for piece in self.game.board.get_all_pieces():
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            board_msg.pieces.append(piece_msg)
        self.board_publisher.publish(board_msg)

    def publish_move_state(self, new_move, new_piece, removed):
        move_msg = Move()
        move_msg.target_row = new_move[0]
        move_msg.target_col = new_move[1]

        piece_msg = Piece()
        piece_msg.row = new_piece.row
        piece_msg.col = new_piece.col
        piece_msg.color = self.color_to_string(new_piece.color)
        piece_msg.king = new_piece.king
        move_msg.piece_for_moving = piece_msg
        
        for piece in removed:
            piece_msg = Piece()
            piece_msg.row = piece.row
            piece_msg.col = piece.col
            piece_msg.color = self.color_to_string(piece.color)
            piece_msg.king = piece.king
            move_msg.removed_pieces.append(piece_msg)

        self.move_publisher.publish(move_msg)

    def publish_hand_detected_state(self):
        hand_detected_msg = HandDetected()
        hand_detected_msg.hand_detected = True
        # self.hand_detected_publisher.publish(hand_detected_msg)

    def _analyze_board_changes(self, new_board_detected):
        """
        Analyze what changed between the old board state and new detected state
        to provide helpful feedback about the attempted move
        """
        if not self.game.board.board or not self.game.board.isBoardCreated:
            print("\n  → Board not initialized yet")
            return
        
        old_board = self.game.board.board
        
        # PRINT CURRENT BOARD STATE FOR DEBUGGING
        self._print_board_state(old_board, "CURRENT BOARD IN MEMORY")
        self._print_board_detection(new_board_detected, "DETECTED BOARD FROM CAMERA")
        
        # Find pieces that disappeared (potential source positions)
        disappeared_pieces = []
        for row in range(8):
            for col in range(8):
                old_piece = old_board[row][col]
                new_piece = new_board_detected[row][col]
                
                # Piece was there, now it's gone
                if old_piece != 0 and new_piece == 0:
                    position = row * 8 + col
                    disappeared_pieces.append((row, col, position, old_piece))
        
        # Find pieces that appeared (potential destination positions)
        appeared_pieces = []
        for row in range(8):
            for col in range(8):
                old_piece = old_board[row][col]
                new_piece = new_board_detected[row][col]
                
                # Empty before, piece now
                if old_piece == 0 and new_piece != 0:
                    position = row * 8 + col
                    color = "WHITE" if new_piece == 1 else "BLACK"
                    appeared_pieces.append((row, col, position, color))
        
        print("\n  MOVE ANALYSIS:")
        
        if len(disappeared_pieces) == 0 and len(appeared_pieces) == 0:
            print("    ⚠ No changes detected on board")
            print("    → Board comparison:")
            
            # Show both boards for debugging
            old_white = sum(1 for row in old_board for piece in row if piece != 0 and hasattr(piece, 'color') and piece.color == WHITE)
            old_black = sum(1 for row in old_board for piece in row if piece != 0 and hasattr(piece, 'color') and piece.color == BLACK)
            new_white = np.count_nonzero(new_board_detected == 1)
            new_black = np.count_nonzero(new_board_detected == 2)
            
            print(f"      Old board: WHITE={old_white}, BLACK={old_black}")
            print(f"      New board: WHITE={new_white}, BLACK={new_black}")
            print("    → This might mean board states are identical or detection failed")
            return
        
        # Show what disappeared
        if len(disappeared_pieces) > 0:
            print(f"    Pieces removed from board:")
            for row, col, pos, piece in disappeared_pieces:
                color = "WHITE" if piece.color == WHITE else "BLACK"
                
                # Get valid moves for this piece BEFORE it was moved
                valid_moves = self.game.board.get_valid_moves(piece)
                move_positions = [move[0] * 8 + move[1] for move in valid_moves.keys()]
                
                print(f"      • {color} piece at position {pos} (row {row}, col {col})")
                if len(move_positions) > 0:
                    print(f"        Valid moves were: {move_positions}")
                else:
                    print(f"        ⚠ This piece had NO valid moves!")
        
        # Show what appeared
        if len(appeared_pieces) > 0:
            print(f"    Pieces added to board:")
            for row, col, pos, color in appeared_pieces:
                print(f"      • {color} piece at position {pos} (row {row}, col {col})")
        
        # Try to match source → destination
        if len(disappeared_pieces) == 1 and len(appeared_pieces) == 1:
            src_row, src_col, src_pos, src_piece = disappeared_pieces[0]
            dst_row, dst_col, dst_pos, dst_color = appeared_pieces[0]
            
            # Check if colors match
            src_color = "WHITE" if src_piece.color == WHITE else "BLACK"
            
            if src_color == dst_color:
                print(f"\n    ATTEMPTED MOVE:")
                print(f"      {src_color} from position {src_pos} → position {dst_pos}")
                print(f"      From (row {src_row}, col {src_col}) → (row {dst_row}, col {dst_col})")
                
                # Check if this destination was valid
                valid_moves = self.game.board.get_valid_moves(src_piece)
                valid_destinations = list(valid_moves.keys())
                
                if (dst_row, dst_col) in valid_destinations:
                    print(f"      ✓ Destination WAS in valid moves!")
                    print(f"      → But the move was still rejected - check game logic")
                else:
                    print(f"      ✗ Destination NOT in valid moves!")
                    move_positions = [move[0] * 8 + move[1] for move in valid_moves.keys()]
                    print(f"      → Valid moves were: {move_positions}")
            else:
                print(f"\n    ⚠ Color mismatch: {src_color} piece removed, {dst_color} piece added")
        
        elif len(disappeared_pieces) > 1:
            print(f"\n    ⚠ Multiple pieces removed ({len(disappeared_pieces)}) - possible capture or hand blocking view")
        
        elif len(appeared_pieces) > 1:
            print(f"\n    ⚠ Multiple pieces added ({len(appeared_pieces)}) - possible detection error")

    def _print_board_state(self, board, title):
        """Print the board state from game memory (Piece objects)"""
        print(f"\n  {title}:")
        print("  " + "="*65)
        print("      0   1   2   3   4   5   6   7")
        print("    " + "-"*65)
        
        for row in range(8):
            row_str = f"  {row} |"
            for col in range(8):
                piece = board[row][col]
                position = row * 8 + col
                
                if piece == 0:
                    row_str += f" {position:2d} |"
                elif hasattr(piece, 'color'):
                    if piece.color == WHITE:
                        row_str += f" W{position:2d}|"
                    else:
                        row_str += f" B{position:2d}|"
                else:
                    row_str += f" ?{position:2d}|"
            print(row_str)
        
        print("    " + "-"*65)
        print("  Legend: W=WHITE (you), B=BLACK (AI), ##=empty position")
        
        # Print piece positions summary
        white_positions = []
        black_positions = []
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0 and hasattr(piece, 'color'):
                    position = row * 8 + col
                    if piece.color == WHITE:
                        white_positions.append(position)
                    else:
                        black_positions.append(position)
        
        print(f"\n  WHITE pieces at: {white_positions}")
        print(f"  BLACK pieces at: {black_positions}")
        print("  " + "="*65)

    def _print_board_detection(self, board_detected, title):
        """Print the detected board state (numeric values: 0=empty, 1=white, 2=black)"""
        print(f"\n  {title}:")
        print("  " + "="*65)
        print("      0   1   2   3   4   5   6   7")
        print("    " + "-"*65)
        
        for row in range(8):
            row_str = f"  {row} |"
            for col in range(8):
                value = board_detected[row][col]
                position = row * 8 + col
                
                if value == 0:
                    row_str += f" {position:2d} |"
                elif value == 1:
                    row_str += f" W{position:2d}|"
                elif value == 2:
                    row_str += f" B{position:2d}|"
                else:
                    row_str += f" ?{position:2d}|"
            print(row_str)
        
        print("    " + "-"*65)
        print("  Legend: W=WHITE (you), B=BLACK (AI), ##=empty position")
        
        # Print piece positions summary
        white_positions = []
        black_positions = []
        for row in range(8):
            for col in range(8):
                value = board_detected[row][col]
                position = row * 8 + col
                if value == 1:
                    white_positions.append(position)
                elif value == 2:
                    black_positions.append(position)
        
        print(f"\n  WHITE pieces at: {white_positions}")
        print(f"  BLACK pieces at: {black_positions}")
        print("  " + "="*65)

def main(args=None):
    rclpy.init(args=args)
    checkers_node = CheckersNode()
    rclpy.spin(checkers_node)

    checkers_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()