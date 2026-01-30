"""
Data Collection Script
RC 트랙 이미지 수집
"""
import cv2
import os
import argparse
from datetime import datetime
from pathlib import Path


class DataCollector:
    """
    RC 트랙 이미지 수집기
    
    수집 전략:
        - 다양한 조명 (밝음/어두움/그림자)
        - 다양한 거리 (0.5m ~ 3m)
        - 다양한 각도 (정면/측면/대각)
        - 다양한 객체 배치 (1~10개)
    
    키 바인딩:
        SPACE: 이미지 저장
        'p': 일시정지/재개
        'q': 종료
        'i': 정보 표시
    """
    
    def __init__(self, output_dir: str = 'datasets/raw', camera_id: int = 0):
        """
        Parameters:
            output_dir: Output directory for collected images
            camera_id: Camera device ID (0 for default)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 카메라 초기화
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # 해상도 설정 (Pi Camera V2: 640×480)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 통계
        self.count = 0
        self.session_start = datetime.now()
    
    def collect_images(self, target: int = 100):
        """
        이미지 수집 (인터랙티브)
        
        Parameters:
            target: Target number of images
        """
        print("\n" + "="*80)
        print("📸 Data Collection Session Started")
        print("="*80)
        print(f"Target:      {target} images")
        print(f"Output dir:  {self.output_dir}")
        print("\nControls:")
        print("  SPACE: Save image")
        print("  'p':   Pause/Resume")
        print("  'i':   Show info")
        print("  'q':   Quit")
        print("="*80 + "\n")
        
        paused = False
        show_info = True
        
        while self.count < target:
            if not paused:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # 정보 오버레이
                if show_info:
                    self._draw_info(frame, self.count, target, paused)
                
                # 프레임 표시
                cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE: 저장
                if not paused:
                    self._save_image(frame)
            
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            
            elif key == ord('i'):  # Toggle info
                show_info = not show_info
            
            elif key == ord('q'):  # Quit
                print("\n🛑 Quit requested")
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 세션 요약
        duration = (datetime.now() - self.session_start).total_seconds()
        
        print("\n" + "="*80)
        print("📊 Collection Session Summary")
        print("="*80)
        print(f"Images collected: {self.count}")
        print(f"Duration:         {duration:.1f} seconds")
        print(f"Output dir:       {self.output_dir}")
        print("="*80 + "\n")
    
    def _save_image(self, frame: np.ndarray):
        """이미지 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{self.count:04d}_{timestamp}.jpg"
        filepath = self.output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        self.count += 1
        
        print(f"✅ [{self.count:4d}] Saved: {filename}")
    
    def _draw_info(self, frame: np.ndarray, count: int, target: int, paused: bool):
        """정보 오버레이 그리기"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w-10, 120), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # 텍스트
        progress = count / target * 100
        
        cv2.putText(frame, f"Count: {count}/{target} ({progress:.1f}%)", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Status: {'PAUSED' if paused else 'RECORDING'}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 165, 255) if paused else (0, 255, 0), 2)
        
        cv2.putText(frame, "SPACE: Save | P: Pause | Q: Quit", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Data Collection for RC Track')
    
    parser.add_argument('--output', type=str, default='datasets/raw',
                       help='Output directory')
    parser.add_argument('--target', type=int, default=100,
                       help='Target number of images')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    
    # 수집 시작
    collector = DataCollector(
        output_dir=args.output,
        camera_id=args.camera
    )
    
    try:
        collector.collect_images(target=args.target)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
    finally:
        print("👋 Data collection finished")


if __name__ == '__main__':
    main()
