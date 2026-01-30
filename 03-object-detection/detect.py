"""
YOLOv8 Inference Script
단일 이미지 또는 비디오에서 객체 감지
"""
import argparse
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.detector import ObjectDetector


def detect_image(
    image_path: str,
    weights: str = 'yolov8l.pt',
    conf_thres: float = 0.25,
    save_result: bool = True,
    output_dir: str = 'results/detections'
):
    """
    단일 이미지 감지
    
    Parameters:
        image_path: Input image path
        weights: Model weights
        conf_thres: Confidence threshold
        save_result: Save annotated image
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"🔍 Object Detection - Single Image")
    print(f"{'='*80}")
    print(f"Image:       {image_path}")
    print(f"Weights:     {weights}")
    print(f"Confidence:  {conf_thres}")
    print(f"{'='*80}\n")
    
    # Detector 초기화
    detector = ObjectDetector(
        weights=weights,
        conf_thres=conf_thres
    )
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 감지
    result = detector.detect(image_rgb, return_image=True)
    
    # 결과 출력
    print(f"✅ Detection Complete!")
    print(f"   Detections:     {result['num_detections']}")
    print(f"   Inference time: {result['inference_time_ms']:.2f} ms")
    print(f"   FPS:            {1000/result['inference_time_ms']:.1f}\n")
    
    if result['num_detections'] > 0:
        print("   Detected objects:")
        for cls_name, conf, box in zip(
            result['class_names'], 
            result['confidences'],
            result['boxes']
        ):
            print(f"     - {cls_name:15s} (conf: {conf:.3f}) at {box}")
    
    # 결과 저장
    if save_result and 'image_annotated' in result:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"detected_{Path(image_path).name}"
        annotated_bgr = cv2.cvtColor(result['image_annotated'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_file), annotated_bgr)
        
        print(f"\n💾 Result saved: {output_file}")
    
    # 시각화
    if 'image_annotated' in result:
        cv2.imshow('Detection Result', 
                   cv2.cvtColor(result['image_annotated'], cv2.COLOR_RGB2BGR))
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_video(
    video_path: str,
    weights: str = 'yolov8l.pt',
    conf_thres: float = 0.25,
    save_result: bool = True,
    output_dir: str = 'results/videos'
):
    """
    비디오 스트림 감지
    
    Parameters:
        video_path: Video file path or 0 for webcam
        weights: Model weights
        conf_thres: Confidence threshold
        save_result: Save output video
        output_dir: Output directory
    """
    print(f"\n{'='*80}")
    print(f"🎥 Object Detection - Video Stream")
    print(f"{'='*80}")
    print(f"Video:       {video_path}")
    print(f"Weights:     {weights}")
    print(f"{'='*80}\n")
    
    # Detector 초기화
    detector = ObjectDetector(weights=weights, conf_thres=conf_thres)
    
    # 비디오 열기
    if video_path == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # 비디오 속성
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video info: {width}×{height} @ {fps} FPS\n")
    
    # 출력 비디오 (선택적)
    writer = None
    if save_result:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"detected_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (width, height))
    
    frame_count = 0
    
    print("🎬 Processing... (Press 'q' to quit)\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 감지
            result = detector.detect(frame_rgb, return_image=True)
            
            frame_count += 1
            
            # 프레임 표시
            if 'image_annotated' in result:
                annotated_bgr = cv2.cvtColor(result['image_annotated'], cv2.COLOR_RGB2BGR)
                
                # FPS 표시
                fps_text = f"FPS: {1000/result['inference_time_ms']:.1f}"
                cv2.putText(annotated_bgr, fps_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                cv2.imshow('Detection', annotated_bgr)
                
                if writer is not None:
                    writer.write(annotated_bgr)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    
    # 통계
    stats = detector.get_performance_stats()
    
    print(f"\n{'='*80}")
    print("📊 Video Detection Stats")
    print(f"{'='*80}")
    print(f"Total frames:       {frame_count}")
    print(f"Avg inference time: {stats['avg_inference_time_ms']:.2f} ms")
    print(f"Avg FPS:            {stats['avg_fps']:.1f}")
    print(f"Avg detections:     {stats['avg_detections']:.1f}")
    print(f"{'='*80}")
    
    if save_result:
        print(f"\n💾 Output saved: {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 Object Detection')
    
    parser.add_argument('--source', type=str, required=True,
                       help='Image or video path (or "0" for webcam)')
    parser.add_argument('--weights', type=str, default='yolov8l.pt',
                       help='Model weights path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--save', action='store_true',
                       help='Save detection results')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # 이미지 vs 비디오 판별
    source_path = Path(args.source) if args.source != '0' else None
    
    if source_path and source_path.is_file():
        # 확장자로 판별
        if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 이미지
            detect_image(
                image_path=args.source,
                weights=args.weights,
                conf_thres=args.conf,
                save_result=args.save,
                output_dir=args.output + '/images'
            )
        else:
            # 비디오
            detect_video(
                video_path=args.source,
                weights=args.weights,
                conf_thres=args.conf,
                save_result=args.save,
                output_dir=args.output + '/videos'
            )
    else:
        # 웹캠
        detect_video(
            video_path=args.source,
            weights=args.weights,
            conf_thres=args.conf,
            save_result=args.save,
            output_dir=args.output + '/videos'
        )


if __name__ == '__main__':
    main()
