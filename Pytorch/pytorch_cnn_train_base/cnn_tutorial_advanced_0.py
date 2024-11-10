import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import argparse
import pprint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime



# argparse 함수 설정
def parse_args():
    parser = argparse.ArgumentParser(description="Simple CNN arguments")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed settings")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="learning rate")
    parser.add_argument("-e", "--epoch", type=int, default=2, help="epoch")
    parser.add_argument("-c", "--checkpoint", type=str, default="./checkpoints", help="checkpoints path")
    parser.add_argument("-l", "--lastname", type=str, default="checkpoint_last_epoch", help="last epoch name")
    parser.add_argument("-r", "--resume", type=str, default="", help="pretrained weight")

    return parser.parse_args()

# seed 함수 설정
def set_seed(seed):
    # Python의 random 모듈 시드 설정
    random.seed(seed)
    
    # NumPy의 시드 설정
    np.random.seed(seed)
    
    # PyTorch의 시드 설정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 환경에서 모든 GPU에 시드 설정
    
    # 연산의 결정성을 위해 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# CNN 모델 정의
# 2개의 cnn block(conv - batch - act - dropout - pool 순서로 정의)
# 마지막은 fc 에는(dropout 추가)

# CNNBlock 클래스 정의
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same", dropout_rate=0.5):
        super(CNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)

# SimpleCNN 클래스 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.layer1 = CNNBlock(3, 16)
        self.layer2 = CNNBlock(16, 32)

        self.fc = nn.Linear(32*8*8, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 32*8*8)
        x = self.fc(x)

        return x



# 학습 함수 정의
def train(net, train_dataloader, optimizer, criterion, device, epoch, epochs):
    # 모드 정의
    net.train()

    # 초기 정의
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_dataloader, desc=f"Training - epoch:[{epoch+1}/{epochs}]", leave=True, ascii=True)
    for images, labels in progress_bar:
        # 데이터 선언 및 device 선택
        images = images.to(device)
        labels = labels.to(device)

        # grad 초기화
        optimizer.zero_grad()

        # inference
        pred = net(images)

        # loss 계산
        loss = criterion(pred, labels)

        # max indices 구하기
        _, predicted = torch.max(pred, 1) # values, indices return
        total += labels.size(0)

        # 맞은 개수 계산
        correct += (predicted == labels).sum().item()

        # 역전파 계산
        loss.backward()

        # 파라미터 업데이트
        optimizer.step()
        
        # loss 계산
        running_loss += loss.item()
        avg_loss = running_loss / (progress_bar.n + 1)
        
        progress_bar.set_postfix(loss="{:.5f}".format(avg_loss))  # loss 값을 4자리 고정
    
    # 정확도 계산
    accuracy = 100 * correct / total

    return avg_loss, accuracy

# 검증 함수 정의
def validate(model, loader, criterion, device, desc):
    # 모드 설정
    model.eval()

    # 초기 세팅
    running_loss = 0.0
    correct = 0
    total = 0

    # no grad 설정
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=True, ascii=True)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # inference
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, labels)

            # loss 합산
            running_loss += loss.item()

            # max indices 
            _, predicted = torch.max(outputs, 1)

            # 전체 image 개수 합산
            total += labels.size(0)

            # 맞은 개수 힙산
            correct += (predicted == labels).sum().item()
    
    # 정확도 계산
    accuracy = 100 * correct / total
    return running_loss / len(loader), accuracy

def main():
    # UTC 시간 출력
    utc_time = datetime.now()
    utc_time = int(utc_time.timestamp())

    args = parse_args()
    # 0. random seed 설정
    set_seed(args.seed)
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    CHECKPOINT_DIR = os.path.join(args.checkpoint, f"{utc_time}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    FINAL_MODEL_PATH = os.path.join(args.checkpoint, f"{utc_time}", args.lastname + ".pth")
    RESUME = args.resume

    


    pprint.pprint(vars(args))
    print()

    # 장비설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 데이터셋 로드
    train_valid_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    print(len(train_valid_dataset))

    # 데이터셋 분할 (train/valid)
    train_size = int(0.8 * len(train_valid_dataset))  # 80%를 학습용
    valid_size = len(train_valid_dataset) - train_size  # 20%를 검증용
    train_dataset, valid_dataset = random_split(train_valid_dataset, [train_size, valid_size])

    # dataloader 선언
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 네트워크 선언
    net = SimpleCNN().to(device)
    print(net)
    print()
    summary(net, (3, 32, 32))

    # state_dict()를 사용해 모델 구조와 파라미터 출력
    print("Model structure and parameters:")
    for param_name, param_tensor in net.state_dict().items():
        print(f"Layer: {param_name} | Size: {param_tensor.size()}")  

    # loss & opimizer 함수 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # 스케쥴러 정의
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    start_epoch = 0
    if RESUME and os.path.exists(RESUME):
        # 체크포인트 로드
        checkpoint = torch.load(os.path.normpath(RESUME))
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # 스케줄러 상태 로드
        start_epoch = checkpoint['epoch'] + 1  # 이어서 학습할 에포크 (이전 에포크의 다음 에포크부터 시작)

        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        print(f"load checkpoint - start_epoch: {start_epoch+1}, train_loss: {train_loss}, valid_loss: {valid_loss}")
        
    
    end_epoch = start_epoch+EPOCHS
    print(f"Start training from {start_epoch+1} to {end_epoch} epochs")
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_accuracy = train(net, train_dataloader, optimizer, criterion, device, epoch, end_epoch)
        valid_loss, valid_accuracy = validate(net, valid_dataloader, criterion, device, "Validating")
        scheduler.step(valid_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{end_epoch}], "
            f"- Learning Rate: {current_lr:.6f}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Train Accuracy: {train_accuracy:.2f}% "
            f"Valid Loss: {valid_loss:.6f}, "
            f"Valid Accuracy: {valid_accuracy:.2f}%")

        # 에포크별 체크포인트 저장
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # 스케줄러 상태 저장
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        print()
        # 최종 모델 저장
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 스케줄러 상태 저장
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, FINAL_MODEL_PATH)
    print(f"Final model saved at {FINAL_MODEL_PATH}")


    test_loss, test_accuracy = validate(net, test_loader, criterion, device, "Testing")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")





if __name__=="__main__":
    main()