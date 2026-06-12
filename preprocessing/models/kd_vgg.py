import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
EOFcat << 'EOF' > preprocessing/models/kd_vgg.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
EOFcat << 'EOF' > preprocessing/models/kd_vgg.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
EOFcat << 'EOF' > preprocessing/models/kd_vgg.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
EOFcat << 'EOF' > preprocessing/models/kd_vgg.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
EOFcat << 'EOF' > preprocessing/models/kd_vgg.py
import torch
import torch.nn as nn
import torchvision.models as models

class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.init_conv = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(*list(vgg.features)[2:5])
        self.stage2 = nn.Sequential(*list(vgg.features)[5:10])
        self.stage3 = nn.Sequential(*list(vgg.features)[10:17])
        self.stage4 = nn.Sequential(*list(vgg.features)[17:24])

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class StudentVGG(nn.Module):
    def __init__(self):
        super(StudentVGG, self).__init__()
        self.init_conv = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.stage1 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        feat1 = self.init_conv(x)
        feat2 = self.stage1(feat1)
        feat3 = self.stage2(feat2)
        feat4 = self.stage3(feat3)
        feat5 = self.stage4(feat4)
        return [feat2, feat3, feat4, feat5]

class KDVGGLoss(nn.Module):
    def __init__(self):
        super(KDVGGLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_feats, teacher_feats):
        loss = 0.0
        for sf, tf in zip(student_feats, teacher_feats):
            if sf.shape != tf.shape:
                proj = nn.functional.interpolate(sf, size=tf.shape[2:], mode='bilinear', align_corners=False)
                loss += self.mse(proj, tf[:, :sf.shape[1], :, :])
            else:
                loss += self.mse(sf, tf)
        return loss

if __name__ == "__main__":
    dummy_input = torch.randn(1, 6, 256, 256)
    teacher = TeacherVGG()
    student = StudentVGG()
    t_feats = teacher(dummy_input)
    s_feats = student(dummy_input)
    print("Local Check Successful:")
    print(f"Teacher outputs: {[f.shape for f in t_feats]}")
    print(f"Student outputs: {[f.shape for f in s_feats]}")
