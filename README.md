IntroGPT that user can ask questions like "Introduce the person" about ingested resume data and the bot will give answers according to the context.
This IntroGPT is an end-to-end prototype of a Generative AI app that introduces yourself to users. Users can ask the app questions and receive pertinent answers. It is built on top of the open-source initiative, localGPT from PromptEngineer.

### For the architectural diagram of the app and detailed explanation of design and insights , please refer to the powerpoint slides under ResumeGPT/Powerpoint Slides. For the video demonstration, it is under ResumeGPT/Video_Demo
  
- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store.
- `run_resumeGPT.py` uses a local LLM to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.

# Environment Setup

1. 📥 Clone the repo using git:

```shell
git clone https://github.com/anabellechan/ResumeGPT-Anabelle.git
```

2. Conda Install for virtual environment management. Create and activate a new virtual environment.

On VSC, command palette -> select interpreter -> select anaconda3 -> e.g Python 3.10.0
```shell
conda create -n resumeGPT python=3.10.0
conda activate resumeGPT
```

3. 🛠️ Install the dependencies using pip

To set up your environment to run the code, first install all requirements:

```shell
pip install -r requirements.txt
```
python3 -m pip install torch torchvision if it says torch is not installed.
pip install llama-cpp-python==0.1.83

Download NVIDIA cuda toolkit 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive

For `NVIDIA` GPUs support, use `cuBLAS`

For Llama-cpp-python,
```shell
# Example: cuBLAS
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

For Apple Metal (`M1/M2`) support, use

```shell
# Example: METAL
CMAKE_ARGS="-DLLAMA_METAL=on"  FORCE_CMAKE=1 pip install llama-cpp-python==0.1.83 --no-cache-dir
```

```shell
DOCUMENT_MAP = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}
```

### Ingest

Run the following command to ingest all the data.

If you have `cuda` setup on your system.
Put resume files into the `SOURCE_DOCUMENTS` folder. Run python ingest.py to ingest the resume.
```shell
python ingest.py 
```
and you are ready to start the program.

Use the device type argument to specify a given device.
To run on `cpu`
```sh
python ingest.py --device_type cpu
```
To run on `M1/M2`

```sh
python ingest.py --device_type mps
```

This will create a new folder called `DB` and use it for the newly created vector store. You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `DB` and reingest your documents.


In order to chat with your documents, run the following command (by default, it will run on `cuda`).

```shell
python run_resumeGPT.py
```
You can also specify the device type just like `ingest.py`

```shell
python run_resumeGPT.py --device_type mps # to run on Apple silicon
```

This will load the ingested vector store and embedding model. You will be presented with a prompt:

```shell
> Enter a query:
```
Once the answer is generated, you can then ask another question without re-running the script, just wait for the prompt again.

Type `exit` to finish the script.

```shell
python run_resumeGPT.py --use_history
```

# Run the Graphical User Interface
The default models used are:
   ```shell
   MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
   MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"
   ```
Navigate to the `/resumeGPT` directory.
Run the following command `python run_resumeGPT_API.py`. Wait for the API to run.

Wait until everything has loaded in. You should see something like `INFO:werkzeug:Press CTRL+C to quit`.

Open up a second terminal and activate the same python environment.

Navigate to the `/ResumeGPT/ResumeGPTUI` directory.

Run the command `python ResumeGPTUI.py`.

Open up a web browser and go the address `http://localhost:5111/`.

# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   - Universal Windows Platform development
   - C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.

# Common Errors

 - [Torch not compatible with CUDA enabled](https://github.com/pytorch/pytorch/issues/30664)

   -  Get CUDA version
      ```shell
      nvcc --version
      ```
      ```shell
      nvidia-smi
      ```
   - Try installing PyTorch depending on your CUDA version
      ```shell
         conda install -c pytorch torchvision cudatoolkit=10.1 pytorch
      ```
   - If it doesn't work, try reinstalling
      ```shell
         pip uninstall torch
         pip cache purge
         pip install torch -f https://download.pytorch.org/whl/torch_stable.html
      ```
    If errors in building llama-cpp-python==0.1.83,  
        download cmake tools in visual studio 2022  
        then download llama-cpp-python==0.1.83  
        pip install --force-reinstall charset-normalizer==3.1.0  

  ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
  ```shell
     pip install h5py
     pip install typing-extensions
     pip install wheel
  ```
- Failed to import transformers
  - Try re-install
    ```shell
       conda uninstall tokenizers, transformers
       pip install transformers
    ```
