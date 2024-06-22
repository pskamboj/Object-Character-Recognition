import stow
import os
from tqdm import tqdm
from configs import ModelConfigs
import tensorflow as tf

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer , LabelIndexer, LabelPadding,ImageShowCV2
from mltu.augmentors import RandomBrightness,RandomRotate,RandomErodeDilate,RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx,TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model 
dataset_path = "E:\Ronak\OCR-1"

dataset , vocab , max_len = [] , set() , 0

words = open(stow.join(dataset_path,"words.txt"),"r").readlines()
for line_num,line in enumerate(tqdm(words)):
    if line.startswith("#"):
        continue
    
    line_split = line.split(" ")
    if len(line_split) < 2:
        print("Error at line", line_num, ": Line doesn't contain enough elements")
        continue
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')

    rel_path = stow.join(dataset_path , "words" , folder1 , folder2 , file_name)
    if not stow.exists(rel_path):
        continue

    dataset.append([rel_path , label])
    vocab.update(list(label))
    max_len = max(max_len , len(label))

    configs = ModelConfigs()

    configs.vocab = "".join(vocab)
    configs.max_text_length=max_len
    configs.save()

    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(configs.width , configs.height,keep_aspect_ratio=False),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length = configs.max_text_length , padding_value=len(configs.vocab)),
        ],
    )
#splitting into training and validation
    train_data_provider , val_data_provider = data_provider.split(split = 0.9)
#augmentation with random brightness,rotation etc
    train_data_provider.augmentors=[
        RandomBrightness(),
        RandomErodeDilate(),
        RandomSharpen(),
        RandomRotate(angle=10),
    ]

#tensorflow model architecture
    model=train_model(
        input_dim=(configs.height,configs.width,3),
        output_dim=len(configs.vocab),
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
)
model.summary(line_length=110)

#callbacks

earlystopper=EarlyStopping(monitor="val_CER",patience=20,verbose=1)
checkpoint=ModelCheckpoint(f"{configs.model_path}/model.h5",monitor="val_CER",verbose=1,save_best_oonly=True, mode="min")
trainLogger=TrainLogger(configs.model_path)
tb_callback=TensorBoard(f"{configs.model_path}/logs",update_freq=1)
reduceLROnpPlat=ReduceLROnPlateau(monitor="val_CER",factor=0.9,min_delta=1e-10,patience=10,verbose=1,mode="auto")
model2onnx=Model2onnx(f"{configs.model_path}/model.h5")

#training

model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper,checkpoint,trainLogger,reduceLROnpPlat,tb_callback,model2onnx],
    workers=configs.train_workers
)
#saving datasets as csv files
train_data_provider.to_csv(os.path.join(configs.model_path,"train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path,"val.csv"))
