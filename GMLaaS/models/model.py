from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self,model_name):
        self.model_name = model_name

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def inference_model(self):
        pass

    @abstractmethod
    def sampling_method(self):
        pass

    @abstractmethod
    def load_data(self):
        pass
