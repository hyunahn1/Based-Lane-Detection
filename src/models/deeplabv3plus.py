"""
DeepLabV3+ 모델 (ResNet101 backbone)
torchvision의 사전학습 모델 활용
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from torchvision.models.segmentation import deeplabv3_resnet101


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for Lane Segmentation
    
    - Backbone: ResNet101 (ImageNet pretrained)
    - Output: Binary segmentation (lane vs background)
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Parameters:
        -----------
        num_classes : int
            출력 클래스 수 (2 = background + lane)
        pretrained : bool
            ImageNet 사전학습 가중치 사용 여부
        """
        super().__init__()
        
        if pretrained:
            # 사전학습 모델 로드 (21 classes)
            self.model = deeplabv3_resnet101(weights='DEFAULT')
            
            # 마지막 classifier만 교체 (Transfer Learning)
            # DeepLabV3의 classifier는 ASPP 모듈 내부에 있음
            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            
            # Auxiliary classifier도 교체
            if hasattr(self.model, 'aux_classifier'):
                aux_in_channels = self.model.aux_classifier[4].in_channels
                self.model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
        else:
            # 사전학습 없이 처음부터 학습
            self.model = deeplabv3_resnet101(weights=None, num_classes=num_classes)
        
        print(f"✅ DeepLabV3+ (ResNet101) initialized")
        print(f"   Pretrained: {pretrained}")
        print(f"   Num classes: {num_classes}")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Total params: {total_params:,}")
        print(f"   Trainable params: {trainable_params:,}")
    
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor, (B, 3, H, W)
        
        Returns:
        --------
        output : torch.Tensor, (B, num_classes, H, W)
        """
        output = self.model(x)['out']
        return output


def get_model(num_classes=2, pretrained=True):
    """모델 생성 헬퍼 함수"""
    return DeepLabV3Plus(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # 테스트
    model = get_model(num_classes=2, pretrained=False)
    
    # Forward test
    x = torch.randn(2, 3, 480, 640)
    y = model(x)
    print(f"\n테스트:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    assert y.shape == (2, 2, 480, 640), f"Output shape mismatch: {y.shape}"
    print("✅ Model test passed!")
