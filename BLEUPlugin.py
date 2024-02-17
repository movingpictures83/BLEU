import nltk

import PyIO
import PyPluMA
class BLEUPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
       self.sentence1 = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["sentence1"])
       self.sentence2 = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["sentence2"])

    def run(self):
        self.BLEUscore = nltk.translate.bleu_score.sentence_bleu([self.sentence2], self.sentence1)

    def output(self, outputfile):
        print(self.BLEUscore)
