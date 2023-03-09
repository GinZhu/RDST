from abc import ABC, abstractmethod

"""
This file provides a basic template of Evaluation. 

An Evaluation function should be:
    1. Callable function
    2. by default it will be called with two parameters:
        1. rec_img: the output of a model
        2. sample: the corresponding sample get from DataLoader;
    3. by default it will return:
        1. reports: a dict of {'metric': [score mean, (score std)]}
        2. records (if necessary): the image-wise score stored in a dict.
    4. .print(report) will return a str. To call this function will help to get the log of evaluation.
    5. .plot(reports/report, plot_dir, prefix): 
        it depends. If quick evaluation, it will plot a series of reports, to show how performances improves (or not).
                    If final evaluation (or testing), it will plot the the scores over the validation / testing dataset.
    6. .save(records, folder, prefix): save the records. Normally will only be used for testing.

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Oct 23 2020
"""


class BasicEvaluation(ABC):

    """
    Callable Abstract Class, a template.
    By implementation, it will be used by a valid Dataset

    By default, it should be called with two params:
        1. rec_img: the output of a network;
        2. sample: a sample from Dataset;

    for image reconstruction related tasks, psnr / ssim are provided as default metrics.

    """

    def __init__(self):
        # the names of all metrics be used
        self.metrics = []

    @abstractmethod
    def __call__(self, rec_img, sample):

        report = {}

        return report

    @abstractmethod
    def display_images(self, rec_img, sample):
        """
        Modify the images, put them in a dict and return to save
        :param rec_img: a list of rec_imgs, the same format as __call__
        :param sample: the same format as in __call__
        :return: a dict, each key means the name while each value is a list of images to display together.

        """
        pass

    def get_metrics(self):
        return self.metrics

    @abstractmethod
    def print(self, report):
        return ''

    def plot_process(self, reports, plot_dir, prefix):
        """
        Plot a series of reports, and save in the folder.
        This function is used to track the validation performance during training.
        :param reports: a list or a tuple of reports
        :param plot_dir: the folder to save figures
        :param prefix: the name of figures
        :return: return a message about saving figures.
        """
        pass

    def plot_final_evas(self, report, plot_dir, prefix):
        """
        Plot the final evaluation report.
        This function is used to display the performance of final evaluation / testing.
        :param report: the evaluation report
        :param plot_dir: the folder to save figures
        :param prefix: the name of figures
        :return: return a message about saving figures.
        """
        pass

    @abstractmethod
    def save(self, reports, folder, prefix):
        """
        To save a report / reports.
        :param reports: the evaluation report
        :param folder: the folder to save figures
        :param prefix: the name of figures
        :return: return a message about saving figures.

        To load:
        1. load validation series:
        repo = np.load(path, allow_pickle=True)

        2. load final validation / testing:
        repo = np.load(path, allow_pickle=True).item()
        """
        pass

    @staticmethod
    def stack_eva_reports(reports):
        # stack each element in eva_report separately
        stacked_report = {}
        for k in reports[0].keys():
            stacked_report[k] = []
            for r in reports:
                stacked_report[k].append(r[k])
        return stacked_report

