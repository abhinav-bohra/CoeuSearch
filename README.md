# CoeuSearch
-----
## _Neural File Search Engine_

![CoeuSearch-Demo](https://github.com/abhinav-bohra/CoeuSearch/blob/main/Documentation/CoeuSearch-Demo.gif)


> In Greek mythology, _Coeus_ was the Titan-god of the inquisitive mind, his name meaning "query" or "questioning"

CoeuSearch is an NLP based intelligent local-file search engine that searches for relevant text documents in a specific folder, considering the semantics of the file’s name & its content and returns the most relevant files.Dillinger is a cloud-enabled, mobile-ready, offline-storage compatible,

#### Input
- Search Directory: Location of folder to be searched
- Search Query: Phrase/keywords to be searched

#### Output
- Relevant Files: Location of top matched files

#### Files Supported
- File Content: .docx, .txt, .pdf, .ppt
- File Name: ALL including images, audio and video files

## Getting Started

Download or clone this repository on your system.

### Prerequisites
```
- PYTHON 3.8.1
- DJANGO 1.1.2
```
### Installing
- Install python3 on your system
- Add python to environment variables
- Navigate to 'CoeuSearch' folder in terminal 
- Create an evironment using the following command -> virtualenv CoeuSearch
- Activate the evironment using the following command  
```
     .\CoeuSearch\Scripts\activate    (For Windows)
      source CoeuSearch\bin\activate  (For Ubuntu)
```
- Run command ```pip3 install -r requirements.txt```
- Run command ```python3 manage.py runserver```
- Click on localhost link generated after execution of previous command

## Future Work
> - Extend this tool to a multi-modal search engine that supports image, audio and video data by utilising Vision Transformers and Automatic Speech Recognition.
> - Enable multilingual search queriesBy by using pre-trained language models like mBERT and XLM
> - Increase inference speed by using Approximate Nearest Neighbors for semantic similarity

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
