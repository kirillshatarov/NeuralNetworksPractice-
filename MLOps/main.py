import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(1 * 28 * 28, 120)
    self.fc2 = nn.Linear(120, 80)
    self.fc3 = nn.Linear(80, 10)

  def forward(self, x):
    x = x.flatten(start_dim=1)  # изменяет форму тензора так, чтобы он имел размерность (batch_size, 28 * 28)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# Загрузка модели и весов
model_path = 'best_model_params.pth'
model = MLP()
best_model_params = torch.load(model_path)
state_dict = best_model_params['model_state_dict']
model.load_state_dict(state_dict)
model.eval()


# Функция для классификации изображения
def classify_digit(image):
    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = transform(image).unsqueeze(0)  # Добавляем размер батча

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return str(predicted.item())


# Интерфейс Gradio
def classify_image(image):
    return classify_digit(image)

inputs = gr.Image(type='pil', label='Загрузите изображение рукописной цифры')
outputs = gr.Textbox(label='Предсказанная цифра')

iface = gr.Interface(
    fn=classify_image,
    inputs=inputs,
    outputs=outputs,
)

iface.launch()

# demo.launch(share=True)












