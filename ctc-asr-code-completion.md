# CTC 语音识别实验代码补全

> 课程实验：语音识别中 CTC 损失函数与 DataLoader 构建

---

## 一、ctc.py — CTC 损失函数模块

### `__init__` 补充（第28行后）

```python
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_loss = nn.CTCLoss(blank=blank_id, reduction='sum', zero_infinity=True)
        self.reduce = reduce
```

### `forward` 补充（第41-46行）

```python
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(self.dropout(hs_pad))

        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1).log_softmax(2)

        # Batch-size average
        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        if self.reduce:
            loss = loss / hs_pad.size(0)
```

---

## 二、dataset_stu.py — 数据加载模块

### TODO 1：拼接音频绝对路径（第24行）

```python
        wav_path = os.path.join(self.data_path, sample['wav'])
```

### TODO 2：torchaudio 读取音频（第28行后）

```python
        waveform, sample_rate = torchaudio.load(wav_path)
```

### TODO 3：kaldi.fbank 提取 Fbank 特征（第32行）

```python
        feat = kaldi.fbank(
            waveform,
            num_mel_bins=self.configs['num_mel_bins'],
            frame_length=self.configs['frame_length'],
            frame_shift=self.configs['frame_shift'],
            sample_frequency=sample_rate,
        )
```

### TODO 4：pad_sequence 补零（第45-46行）

```python
        padded_feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
        feats_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
```

### TODO 5：指定路径并创建 Dataset（第63-68行）

```python
        config_path = 'conf/train.yaml'  # 根据实际路径修改
        with open(config_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)

        train_data_list = 'data/train/data.list'  # 根据实际路径修改
        train_dataset = ASRDataset(train_data_list)
```

### TODO 6：创建 DataLoader（第71行）

```python
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
```

---

## 关键注意点

| 项目 | 说明 |
|------|------|
| `zero_infinity=True` | **必须加**，防止 target 长度 > 输入长度时 loss 变为 inf/NaN |
| `self.configs` | TODO 3 中用 `self.configs` 而非 `configs`，否则 `__getitem__` 找不到变量 |
| 转置 `(B,T,V) → (T,B,V)` | `nn.CTCLoss` 要求输入为 `(T, N, C)` 格式 |
| `reduction='sum'` 配合 `/hs_pad.size(0)` | 先求和再除以 batch_size，得到每样本平均 loss |
