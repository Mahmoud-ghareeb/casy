from dataclasses import dataclass
from typing import ClassVar, Dict

@dataclass
class Args:
    checkpoint_path: str = r"E:\chat\wav2lip_master\checkpoints\wav2lip.pth"
    audio: str = r"E:\chat\inputs\ms.wav"
    face: str = r"E:\chat\inputs\salma720.mp4"
    outfile: str = r'E:\chat\outputs\out.mp4'
    frame_path: str = r"E:\chat\frames"
    fps: int = 25
    face_det_batch_size: int = 16
    wav2lip_batch_size: int = 32
    resize_factor: int = 1
    crop: ClassVar[list[int]] = [0, -1, 0, -1]
    box: ClassVar[list[int]] = [-1, -1, -1, -1]
    rotate: bool = False
    nosmooth: bool = False
    save_frames: bool = False
    static: bool = False
    save_as_video: bool = False
    img_size: int = 96
    pads: ClassVar[list[int]] = [0, 0, 0, 0]
    mel_step_size: int = 16
    device: str = "cpu"
    model_id: ClassVar[dict[str, str]] = {
        "bert-base-multilingual-uncased": "google-bert/bert-base-multilingual-uncased",
        "paraphrase-MiniLM": "models/paraphrase-MiniLM-L3-v2",
        "paraphrase-multilingual-MiniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "all-mpnet-base": "sentence-transformers/all-mpnet-base-v2",
        "multilingual-e5-base": "intfloat/multilingual-e5-base"
    }
    dim: int = 768
    num_of_frames: int = 500
    system_prompt: str = """
            I'll provide you with a JSON object that contains a question and the context related to it:
            {"question": the question, "context": the context}
            Please generate your answer based on the context above.
            If you don't know the answer just say you donot know an do not try to make up the answer nor try to use outside souces to answer.
            """