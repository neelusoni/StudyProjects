from urllib.request import urlopen
from urllib.parse import quote
def read_text():
    test_text = open("/Users/neelu/StudyProjects/test_txt.txt")
    test_text_contents = test_text.read()

    if check_profanity(test_text_contents):
        print ("Profanity detected.Aborting!")
    else:
        print ("File is clean")
        print(test_text_contents)

    test_text.close()

def check_profanity(text_to_check):
    connection = urlopen("http://www.wdylike.appspot.com/?q="+quote(text_to_check))
    profanity_response = connection.read()
    connection.close()
    return profanity_response

read_text()