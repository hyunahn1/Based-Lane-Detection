"""
데이터 품질 체크 스크립트
새 데이터셋의 문제를 빠르게 진단
"""
import json
from pathlib import Path
from PIL import Image
import sys


def check_data_quality():
    """데이터 품질 체크"""
    
    print("\n" + "="*80)
    print("🔍 데이터 품질 체크")
    print("="*80 + "\n")
    
    img_dir = Path('training_data/images')
    ann_dir = Path('training_data/annotations')
    
    # 디렉토리 존재 확인
    if not img_dir.exists():
        print(f"❌ 이미지 폴더가 없습니다: {img_dir}")
        return False
    
    if not ann_dir.exists():
        print(f"❌ 어노테이션 폴더가 없습니다: {ann_dir}")
        return False
    
    # 1. 개수 확인
    print("📁 1. 파일 개수 확인")
    print("-" * 80)
    
    images = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
    jsons = list(ann_dir.glob('*.json'))
    
    print(f"   이미지:      {len(images):>4}개")
    print(f"   어노테이션:  {len(jsons):>4}개")
    
    if len(images) == 0 or len(jsons) == 0:
        print("\n❌ 데이터가 없습니다!")
        return False
    
    if len(images) != len(jsons):
        print(f"\n⚠️  경고: 개수가 다릅니다! (차이: {abs(len(images) - len(jsons))}개)")
    else:
        print("\n✅ 개수 일치")
    
    print()
    
    # 2. 매칭 확인
    print("🔗 2. 파일명 매칭 확인")
    print("-" * 80)
    
    img_names = {p.stem for p in images}
    json_names = {p.stem for p in jsons}
    
    missing_json = img_names - json_names
    missing_img = json_names - img_names
    
    if missing_json:
        print(f"\n⚠️  어노테이션 없는 이미지: {len(missing_json)}개")
        for name in sorted(list(missing_json))[:10]:
            print(f"      - {name}.png")
        if len(missing_json) > 10:
            print(f"      ... 외 {len(missing_json) - 10}개")
    
    if missing_img:
        print(f"\n⚠️  이미지 없는 어노테이션: {len(missing_img)}개")
        for name in sorted(list(missing_img))[:10]:
            print(f"      - {name}.json")
        if len(missing_img) > 10:
            print(f"      ... 외 {len(missing_img) - 10}개")
    
    if not missing_json and not missing_img:
        print("✅ 모든 파일 매칭 완료")
    
    print()
    
    # 3. JSON 포맷 및 내용 확인
    print("📝 3. 어노테이션 품질 확인")
    print("-" * 80)
    
    empty_annotations = []
    invalid_jsons = []
    valid_count = 0
    total_shapes = 0
    
    for json_path in jsons:
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            # 두 가지 포맷 지원
            shapes = data.get('shapes', [])  # LabelMe 포맷
            lanes = data.get('lanes', [])     # 커스텀 포맷
            
            annotations = shapes if shapes else lanes
            
            if not annotations or len(annotations) == 0:
                empty_annotations.append(json_path.name)
            else:
                valid_count += 1
                total_shapes += len(annotations)
        
        except json.JSONDecodeError as e:
            invalid_jsons.append((json_path.name, f"JSON 파싱 에러: {str(e)[:50]}"))
        except Exception as e:
            invalid_jsons.append((json_path.name, f"에러: {str(e)[:50]}"))
    
    print(f"   유효한 어노테이션: {valid_count}개")
    print(f"   평균 shape 개수:   {total_shapes / max(valid_count, 1):.1f}개/이미지")
    
    if empty_annotations:
        print(f"\n⚠️  빈 어노테이션 (차선 없음): {len(empty_annotations)}개")
        for name in sorted(empty_annotations)[:10]:
            print(f"      - {name}")
        if len(empty_annotations) > 10:
            print(f"      ... 외 {len(empty_annotations) - 10}개")
    
    if invalid_jsons:
        print(f"\n❌ 잘못된 JSON: {len(invalid_jsons)}개")
        for name, error in sorted(invalid_jsons)[:10]:
            print(f"      - {name}: {error}")
        if len(invalid_jsons) > 10:
            print(f"      ... 외 {len(invalid_jsons) - 10}개")
    
    if not empty_annotations and not invalid_jsons:
        print("\n✅ 모든 어노테이션 정상")
    
    print()
    
    # 4. 이미지 크기 확인
    print("📐 4. 이미지 크기 확인 (샘플 20개)")
    print("-" * 80)
    
    sizes = {}
    sample_images = list(images)[:20] if len(images) > 20 else images
    
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            size = f"{img.width}x{img.height}"
            sizes[size] = sizes.get(size, 0) + 1
        except Exception as e:
            print(f"⚠️  이미지 읽기 실패: {img_path.name} - {e}")
    
    for size, count in sorted(sizes.items()):
        print(f"   {size:12s}: {count:>3}개")
    
    if len(sizes) > 1:
        print(f"\n⚠️  경고: 여러 크기가 혼재되어 있습니다 ({len(sizes)}종류)")
        print("   → 학습 시 자동으로 리사이즈되므로 괜찮습니다")
    else:
        print("\n✅ 모든 이미지 크기 일치")
    
    print()
    
    # 5. 샘플 JSON 확인
    print("🔬 5. 샘플 JSON 구조 확인")
    print("-" * 80)
    
    if jsons:
        sample_json = jsons[0]
        try:
            with open(sample_json) as f:
                data = json.load(f)
            
            print(f"   파일: {sample_json.name}")
            print(f"   키:   {list(data.keys())}")
            
            # LabelMe 포맷
            if 'shapes' in data and data['shapes']:
                shape = data['shapes'][0]
                print(f"   포맷: LabelMe")
                print(f"   Shape 예시:")
                print(f"      - label: {shape.get('label')}")
                print(f"      - points: {len(shape.get('points', []))}개")
                print(f"      - shape_type: {shape.get('shape_type')}")
                
                required_fields = ['label', 'points', 'shape_type']
                missing = [f for f in required_fields if f not in shape]
                
                if missing:
                    print(f"\n   ⚠️  누락된 필드: {missing}")
                else:
                    print(f"\n   ✅ 필수 필드 모두 존재")
            
            # 커스텀 lanes 포맷
            elif 'lanes' in data and data['lanes']:
                lanes = data['lanes']
                print(f"   포맷: 커스텀 (lanes)")
                print(f"   Lanes 예시:")
                print(f"      - 개수: {len(lanes)}개")
                if lanes and len(lanes[0]) > 0:
                    print(f"      - 첫 번째 lane의 점: {len(lanes[0])}개")
                
                required_fields = ['image', 'width', 'height', 'lanes']
                missing = [f for f in required_fields if f not in data]
                
                if missing:
                    print(f"\n   ⚠️  누락된 필드: {missing}")
                else:
                    print(f"\n   ✅ 필수 필드 모두 존재")
            
        except Exception as e:
            print(f"   ❌ 샘플 읽기 실패: {e}")
    
    print()
    
    # 최종 요약
    print("="*80)
    print("📊 최종 요약")
    print("="*80)
    
    issues = []
    
    if len(images) != len(jsons):
        issues.append(f"개수 불일치 ({len(images)} vs {len(jsons)})")
    
    if missing_json:
        issues.append(f"어노테이션 누락 {len(missing_json)}개")
    
    if missing_img:
        issues.append(f"이미지 누락 {len(missing_img)}개")
    
    if empty_annotations:
        issues.append(f"빈 어노테이션 {len(empty_annotations)}개")
    
    if invalid_jsons:
        issues.append(f"잘못된 JSON {len(invalid_jsons)}개")
    
    if issues:
        print(f"\n⚠️  발견된 문제: {len(issues)}개")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\n💡 권장사항:")
        print("   1. 위 문제들을 수정하세요")
        print("   2. 문제가 심각하면 데이터 재준비를 권장합니다")
        print()
        return False
    else:
        print("\n✅ 데이터셋 품질 양호!")
        print("\n다음 단계:")
        print("   1. python training_data/convert_coco.py")
        print("   2. python src/data/split_data.py")
        print("   3. python train_optimized.py")
        print()
        return True


def main():
    """메인"""
    try:
        success = check_data_quality()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 에러: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
