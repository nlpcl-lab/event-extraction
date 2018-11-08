from bs4 import BeautifulSoup

if __name__ == '__main__':
    """
    To get phrase using START, END offsets
    <extent>
      <charseq START="754" END="793">Secretary of Homeland Security Tom Ridge</charseq>
    </extent>
    """
    with open('./data/ace_2005_td_v7/data/English/bc/adj/CNN_CF_20030303.1900.00.sgm', 'r') as f:
        data = f.read()
        soup = BeautifulSoup(data, features='html.parser')
        text = soup.text
        print(text[754-1:793])
