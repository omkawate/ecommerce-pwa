import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gui.confusion_tools import matrix_counts, folder_label

class ConfusionTests(unittest.TestCase):
    def test_true_rows_predicted_columns(self):
        rows=[{'true_label':'Forest','predicted_class':'River'},
              {'true_label':'River','predicted_class':'River'},
              {'true_label':'Forest','predicted_class':'Forest'}]
        matrix,skipped=matrix_counts(rows,['Forest','River'])
        self.assertEqual(matrix.tolist(),[[1,1],[0,1]])
        self.assertEqual(skipped,0)
    def test_unknown_and_failed_never_count_as_correct(self):
        matrix,skipped=matrix_counts([{'predicted_class':'Forest'}, {'error':'bad file'},
            {'true_label':'Unknown','predicted_class':'Forest'}],['Forest'])
        self.assertEqual(int(matrix.sum()),0)
        self.assertEqual(skipped,3)
    def test_nested_class_folder(self):
        self.assertEqual(folder_label('/data/Forest/subset/image.jpg',['Forest','River']),'Forest')
        self.assertEqual(folder_label('/data/unlabeled/image.jpg',['Forest','River']),'')

if __name__=='__main__':unittest.main()
