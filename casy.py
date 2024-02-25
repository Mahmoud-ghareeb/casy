import os
import sys
import pandas as pd
from typing import Any
from docx import Document
import elevenlabs
import subprocess
import os
from typing import Iterator
from random import randint

from wav2lip_master import inference_yolo

from wav2lip_master import audio

from pydub import AudioSegment
from io import BytesIO
import numpy as np
import cv2

import torch
from ultralytics import YOLO
from IPython.display import display
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

from args import Args



class Casy:
    def __init__(self, file_path, i):

        self.args = Args()

        self.wav2lib_model = inference_yolo.load_model(self.args.checkpoint_path)
        self.yolo_model = YOLO('wav2lip_master/yolo/best.pt')
        
        self.file_path = file_path
        chroma_client = chromadb.PersistentClient(path=f"./dp/demo{i}")
        self.collection = chroma_client.get_or_create_collection(
            name="book",
            metadata={"hnsw:space": "cosine"}
        )
        full_text = self.read_docx(self.file_path)
        splitted_txt = self.splitter(full_text)
        self.model = self._encode()
        encoded_text = self.model.encode(splitted_txt, show_progress_bar=True).tolist()
        ids = [str(i) for i in range(len(encoded_text))]
        self.collection.upsert(
            documents=splitted_txt,
            embeddings=encoded_text,
            ids=ids
        )
        
        api_key = "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"
        elevenlabs.set_api_key("f8b8bd17f45040b85ee67d3d0c6f0b1d")
        gemini_api = "AIzaSyADgF911apMWrew1bvsazxFZOyn5YROLfI"
        
        self.client = OpenAI(api_key=api_key)

    def run(self, question, video, lang="en"):

        self.video = video

        if lang == "en":
            self.messages = [
                {"role": "system", "content": self.args.system_prompt}, 
            ]
        else:
            self.messages = [
                {"role": "system", "content": self.args.system_prompt_ar}, 
            ]

        question_embed = self.model.encode(question)
        results = self.collection.query(
            query_embeddings=question_embed.tolist(),
            n_results=10,  
        )
        top_paragraph = ' '.join([i for i in results['documents']][0])
        prompt = '{"question": ' + question + ', "context": ' + top_paragraph + '}'

        self.messages.append(
            {"role": "user", "content": prompt}
        )
        self.audio = b""
        self.out = cv2.VideoWriter(f"temp/res1.avi",
                                        cv2.VideoWriter_fourcc(*'DIVX'), 25, (720, 720))
        self.cntr = 0
        self.generate_audio(prompt, self.messages)

        if self.video:
            self.out.release()

            print("last frame: ", self.cntr)
            
            audio_segment = AudioSegment.from_file(BytesIO(self.audio), format="mp3")
            audio_segment_resampled = audio_segment.set_frame_rate(16000)
            audio_segment_resampled.export('temp/res.wav', format="wav")
            
            # -c:v copy -c:a aac -strict experimental -ar 16000 -shortest -q:v 1
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format('temp/res.wav', 'temp/res1.avi', 'temp/res.mp4')
            subprocess.call(command, shell=False)

    def _encode(self):
        return SentenceTransformer(self.args.model_id['paraphrase-MiniLM'], device=self.args.device)

    def read_docx(self, file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        full_text = '\n'.join(full_text)

        return full_text

    def splitter(self, txt):
        
        chunk_size = 1000
        chunk_overlap = 200

        def length_function(text: str) -> int:
            return len(text)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )

        return splitter.split_text(txt)

    def face_detect(self, images, args):
        """
        Detect faces in a batch of images using YOLO.
        """
        batch_size = args.face_det_batch_size
        # batch_size = 1
        
        while 1:
            predictions = []
            try:
                for i in range(0, len(images), batch_size):
                    results = self.yolo_model.predict(images[i:i + batch_size], verbose=False, device=self.args.device)
                    try:
                        # boxes = results[0].boxes.xyxy[0].tolist()
                        boxes = results[0].boxes.cpu().xyxy[0].tolist()
                        predictions.append(boxes)
                    except Exception as e:
                        cv2.imwrite(f"temp/faulty_frame{randint(0, 10000)}.jpg", images[0])
                        print("face not detected")
                    
            except RuntimeError:
                if batch_size == 1: 
                    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = args.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
                raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
            
            y1 = max(0, int(rect[1]) - pady1)
            y2 = min(image.shape[0], int(rect[3]) + pady2)
            x1 = max(0, int(rect[0]) - padx1)
            x2 = min(image.shape[1], int(rect[2]) + padx2)
            
            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        return results 

    def datagen(self, mels, args):
        """
        Data generator for processing batches.
        """
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        reader = self.read_frames(self.cntr)
        t = []
        prev = None
        for i, m in enumerate(mels):
            self.cntr += 1
            try:
                frame_to_save = next(reader)
            except StopIteration:
                reader = self.read_frames(self.cntr)
                frame_to_save = next(reader)
            
            try:
                prev = self.face_detect([frame_to_save], args)[0]
                face, coords = prev
            except:
                face, coords = prev

            face = cv2.resize(face, (args.img_size, args.img_size))
                
            if i%10000 == 0:
                cv2.imwrite(f"test{i}.jpg", face)

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= args.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, args.img_size//2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
        
        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch

    def read_frames(self, i):
        """
        Read frames from a folder of image files.
        """

        i %= self.args.num_of_frames
        image_file = f"{i}.jpg"
        image_path = os.path.join(self.args.frame_path, image_file)
        frame = cv2.imread(image_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        yield frame

    def _get_apen_ai_answer(self, prompt, messages):
        response = self.client.chat.completions.create(
            model = "gpt-3.5-turbo-1106",
            temperature= 0,
            messages=messages,
            stream=True
        )
        i = 0
        for chunk in response:
            txt = chunk.choices[0].delta.content
            txt = txt if txt is not None else ""
            print(txt, end="")
            # yield txt if txt is not None else ""
    
    def generate_audio(self, prompt, messages):
        # print(prompt)
        self._get_apen_ai_answer(prompt, messages)
        # audio_bytes = elevenlabs.generate(text=self._get_apen_ai_answer(prompt, messages), voice="Glinda", model="eleven_monolingual_v1", stream=True)
        
        # if not self.video:
        #     self.stream(audio_bytes)
        # else:
        #     cum_chunk = b""
        #     cum_cntr = 0
        #     for chunk in audio_bytes:
        #         if chunk is not None:
        #             if cum_cntr < 2:
        #                 cum_chunk += chunk
        #                 cum_cntr += 1
        #             if cum_cntr == 2: 
        #                 video_bytes = self._process_video(cum_chunk)
        #                 self.stream_frames(video_bytes)
        #                 cum_chunk = b""
        #                 cum_cntr = 0

        #     if cum_chunk: 
        #         video_bytes = self._process_video(cum_chunk)
        #         self.stream_frames(video_bytes)

        #     self.stream(audio_bytes)

    def _process_video(self, chunk):
        
        os.makedirs(f"tests/tst_{self.cntr}", exist_ok=True)

        audio_segment = AudioSegment.from_file(BytesIO(chunk), format="mp3")
        audio_segment.export('temp/temp.mp3', format="mp3")
        audio_segment.export(f'tests/tst_{self.cntr}/temp.mp3', format="mp3")
        command = 'ffmpeg -y -i {} -strict -2 {}'.format('temp/temp.mp3', 'temp/temp.wav')
        subprocess.call(command, shell=True)
        audio_path = 'temp/temp.wav'
        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)

        mel_chunks = []
        mel_idx_multiplier = 80./self.args.fps 
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.args.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.args.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + self.args.mel_step_size])
            i += 1
        print("melchuncks: ", len(mel_chunks))
        vg = len(mel_chunks)
        if vg > 1:
            self.audio += chunk
        
        print("start chunck: ", self.cntr)
        gen = self.datagen(mel_chunks, self.args)
        byte_image = b""
        for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.args.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.args.device)

            with torch.no_grad():
                try:
                    pred = self.wav2lib_model(mel_batch, img_batch)
                except:
                    # print(1)
                    # f = cv2.imread("frames/0.jpg")
                    # self.out.write(f)
                    continue

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            i = 0
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                # self.out.write(f) 
                cv2.imwrite(f'tests/tst_{self.cntr}/{i}.jpg', f)
                frame_bytes = cv2.imencode('.jpg', f)[1].tobytes()
                i += 1
                yield frame_bytes
                # _, encoded_image = cv2.imencode('.png', f) 
                # byte_image += encoded_image.tobytes()
            
                # yield byte_image               
                    
    def stream_frames(self, frame_generator):
       
        for frame_bytes in frame_generator:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            self.out.write(img)

    def stream(self, audio_stream: Iterator[bytes]) -> bytes:

        mpv_command = ["C:\\Program Files\\mpv\\mpv.exe", "--no-cache", "--no-terminal", "--", "fd://0"]
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        audio = b""

        for chunk in audio_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                audio += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return audio
    
    def stream_video(self, video_stream: Iterator[bytes]) -> bytes:

        # ffmpeg_command = [
        #     'ffmpeg',
        #     '-f', 'rawvideo',
        #     '-pixel_format', 'bgr24',
        #     '-video_size', '720x720',
        #     '-i', '-',
        #     '-f', 'mpegts',
        #     '-codec:v', 'mpeg1video',
        #     '-bf', '0',
        #     '-'
        # ]
        # mpv_command = ['|', 'C:\\Program Files\\mpv\\mpv.exe', '--no-cache', '--no-terminal', '--', '-']

        # Combine both commands (simplified for explanation; actual implementation may vary)
        # process_command = ' '.join(ffmpeg_command + mpv_command)
        # process = subprocess.Popen(process_command, stdin=subprocess.PIPE, shell=True)
        mpv_command = ["C:\\Program Files\\mpv\\mpv.exe", "--no-cache", "--no-terminal", "--demuxer=rawvideo", "--demuxer-rawvideo-w=720", "--demuxer-rawvideo-h=720", "--demuxer-rawvideo-fps=25", "--", "-"]
        # mpv_command = [
        #     "C:\\Program Files\\mpv\\mpv.exe",
        #     "--no-cache",
        #     "--no-terminal",
        #     "temp/res.mp4"  # Directly specify the video file path here
        # ]
        # # Replace WIDTH, HEIGHT, and FPS with actual values for your video
        mpv_process = subprocess.Popen(
            mpv_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        video = b""

        for chunk in video_stream:
            if chunk is not None:
                mpv_process.stdin.write(chunk)  # type: ignore
                mpv_process.stdin.flush()  # type: ignore
                video += chunk

        if mpv_process.stdin:
            mpv_process.stdin.close()
        mpv_process.wait()

        return video
