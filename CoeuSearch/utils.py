import re
from io import StringIO
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.converter import TextConverter
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
stop_words = set(stopwords.words('english'))

#---------------------------------------------------------------------------------------------------------------------------
# PDF PARSER
#---------------------------------------------------------------------------------------------------------------------------
def parser_pdf(path):
    output_string = StringIO()
    with open(path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return output_string.getvalue()

#---------------------------------------------------------------------------------------------------------------------------
# CLEAN TEXT AND CONVERT ARRAY OF LINES TO STRING
#---------------------------------------------------------------------------------------------------------------------------
def cleanText(lines):
    if lines and len(lines) > 0:
        lines = [str(l) for l in lines if len(l)>0]
        for i in range(len(lines)):
            word_tokens = word_tokenize(lines[i])
            lines[i] = " ".join([w for w in word_tokens if not w.lower() in stop_words])
        lines = [re.sub(" +", " ",l) for l in lines]
        lines = [re.sub(r'[0-9]', " ",l) for l in lines]
        lines = [l.replace("\n", "") for l in lines]
        text = " ".join(lines)
        return text
    else:
        return None

#---------------------------------------------------------------------------------------------------------------------------
# LOGGING UTILS
#---------------------------------------------------------------------------------------------------------------------------
class Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
