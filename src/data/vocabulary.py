import json
import os

class Vocabulary:
    def __init__(self, pad_token="<PAD>", unk_token="<UNK>", sos_token="<SOS>", eos_token="<EOS>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        
        # Khởi tạo từ điển với các token đặc biệt
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = {}
        
        for token in [pad_token, unk_token, sos_token, eos_token]:
            self.add_word(token)

    def add_word(self, word):
        """Thêm một từ vào từ điển."""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_counts[word] = 1
        else:
            self.word_counts[word] += 1

    def add_sentence(self, sentence):
        """Thêm toàn bộ câu (danh sách các token) vào từ điển."""
        for word in sentence:
            self.add_word(word)

    def get_idx(self, word):
        """Lấy ID của từ, trả về ID của <UNK> nếu từ không tồn tại."""
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def get_word(self, idx):
        """Lấy từ từ ID."""
        return self.idx2word.get(idx, self.unk_token)

    def __len__(self):
        """Trả về kích thước từ điển."""
        return len(self.word2idx)

    def save(self, path):
        """Lưu từ điển thành file JSON."""
        data = {
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "special_tokens": [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path):
        """Tải từ điển từ file JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        specials = data["special_tokens"]
        vocab = cls(specials[0], specials[1], specials[2], specials[3])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        return vocab
    


# # Khởi tạo
# vocab = Vocabulary()

# # Giả sử bạn đọc dữ liệu từ preprocessed_data/rstdt/train
# train_sentences = [["Đây", "là", "câu", "ví", "dụ"], ["Cấu", "trúc", "RST"]]
# for sent in train_sentences:
#     vocab.add_sentence(sent)

# # Lưu lại để dùng sau này
# vocab.save("preprocessed_data/rstdt/vocab.json")

# # Truy xuất
# print(f"ID của từ 'câu': {vocab.get_idx('câu')}")
# print(f"Kích thước từ điển: {len(vocab)}")