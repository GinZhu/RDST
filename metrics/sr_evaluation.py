from metrics.basic_evaluation import BasicEvaluation

import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from os.path import join

from metrics.sr_metrics import SRMetrics


"""
This file provides Evaluations for Super-Resolution tasks. 

Todo:
    1. Add comments
    2. Add examples

@Jin Zhu (jin.zhu@cl.cam.ac.uk) Oct 23 2020
"""


class BasicSREvaluation(BasicEvaluation):

    def __init__(self, metrics, sr_factor=2., gpu_id=-1, record_mode='full'):
        super(BasicSREvaluation, self).__init__()

        self.sr_factor = sr_factor

        modes = ['full', 'mean']
        assert record_mode in modes, 'record mode should be one of {}'.format(modes)
        self.record_mode = record_mode

        self.func = SRMetrics(metrics, gpu_id, self.record_mode)

        self.metrics = self.func.metrics

    def __call__(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        gt_imgs = [s['gt'] for s in samples]

        reports = self.func(gt_imgs, rec_imgs, int(self.sr_factor))
        return reports

    def print(self, report):
        table = []
        row = ['{:.2}'.format(self.sr_factor)]
        for m in self.metrics:
            v = report[m]
            if isinstance(v, (float, int)):
                row += ['{:.4}'.format(v)]
            else:
                mean_v = np.mean(v)
                std_v = np.std(v)
                row += ['{:.4}({:.2})'.format(mean_v, std_v)]
        table.append(row)
        headers = ['SR', ] + self.metrics
        plog = tabulate(table, headers=headers)
        return plog

    def display_images(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        imgs = []
        for r, s in zip(rec_imgs, samples):
            imgs.append(r)
            imgs.append(s['gt'])
        return {'SR x{}'.format(self.sr_factor): imgs}

    def plot_process(self, reports, plot_dir, prefix, step=1):
        assert isinstance(reports, list), 'Input reports must be a list of reports'
        plog = ''
        for m in self.metrics:
            data = {}
            for s in [self.sr_factor]:
                k = '{}'.format(m)
                vs = []
                for r in reports:
                    v = r[k]
                    if isinstance(v, (float, int)) or len(v) == 1:
                        vs.append(v)
                    else:
                        vs.append(np.mean(v))
                data['sr: {:.2}'.format(s)] = vs
            # ## plot
            for k in sorted(data.keys()):
                plt.plot(data[k])
            plt.legend(sorted(data.keys()))
            plt.xlabel('Training Step')
            plt.ylabel(m)
            plt.grid(True)
            plt.xticks(np.arange(len(reports)) * step)
            plt.savefig(join(plot_dir, '{}_{}.png'.format(prefix, m)))
            plt.close()
            plog += 'Figure saved: {}_{}.png\n'.format(prefix, m)
        return plog

    def plot_final_evas(self, report, plot_dir, prefix):
        # for SR tasks with only one specific up-sampling scale, this function is useless (a little)
        pass

    def save(self, reports, folder, prefix):
        np.save(join(folder, '{}.npy'.format(prefix)), reports)
        plog = 'All reports saved to {}'.format(
            join(folder, '{}.npy'.format(prefix))
        )
        return plog


class MetaSREvaluation(BasicEvaluation):
    """
    sr_factors:
        a list of sr scale factors, e.g. [1.5, 2.0, 2.5, 3.0]
    rec_imgs:

    sample:

    """

    def __init__(self, metrics, sr_factors, gpu_id=-1, record_mode='full'):
        super(MetaSREvaluation, self).__init__()
        self.metrics = []
        self.sr_factors = sr_factors

        modes = ['full', 'mean']
        assert record_mode in modes, 'record mode should be one of {}'.format(modes)
        self.record_mode = record_mode

        self.func = SRMetrics(metrics, gpu_id, self.record_mode)

        self.basic_metrics = self.func.metrics
        for m in self.basic_metrics:
            for s in sr_factors:
                self.metrics.append('{}_{}'.format(m, s))

    def __call__(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        report = {}
        for s in self.sr_factors:
            gt_imgs = [sample[s]['gt'] for sample in samples]
            rec_imgs_with_scale = [rec_img[s] for rec_img in rec_imgs]
            report_with_scale = self.func(gt_imgs, rec_imgs_with_scale, int(np.ceil(s)))
            for m in report_with_scale:
                report['{}_{}'.format(m, s)] = report_with_scale[m]

        return report

    def display_images(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        all_imgs = {}
        for s in self.sr_factors:
            imgs = []
            gt_imgs = [sample[s]['gt'] for sample in samples]
            rec_imgs_with_scale = [rec_img[s] for rec_img in rec_imgs]
            for r, g in zip(rec_imgs_with_scale, gt_imgs):
                imgs.append(r)
                imgs.append(g)
            all_imgs['SR x{}'.format(s)] = imgs
        return all_imgs

    def print(self, report):
        table = []
        for s in self.sr_factors:
            row = ['{:.2}'.format(s), ]
            for m in self.basic_metrics:
                v = report['{}_{}'.format(m, s)]
                if isinstance(v, (float, int)):
                    row += ['{:.4}'.format(v)]
                else:
                    if isinstance(v, list) and isinstance(v[0], list):
                        v = np.concatenate(v)
                    mean_v = np.mean(v)
                    std_v = np.std(v)
                    row += ['{:.4}({:.2})'.format(mean_v, std_v)]
            table.append(row)
        headers = ['SR', ] + self.basic_metrics
        plog = tabulate(table, headers=headers)
        return plog

    def plot_process(self, reports, plot_dir, prefix='', step=1):
        plog = ''
        for m in self.basic_metrics:
            data = {}
            for s in self.sr_factors:
                k = '{}_{}'.format(m, s)
                vs = []
                for r in reports:
                    v = r[k]
                    if isinstance(v, (float, int)) or len(v) == 1:
                        vs.append(v)
                    else:
                        vs.append(np.mean(v))
                data['sr: {:.2}'.format(s)] = vs
            # ## plot
            for k in sorted(data.keys()):
                plt.plot(data[k])
            plt.legend(sorted(data.keys()))
            plt.xlabel('Training Step')
            plt.ylabel(m)
            plt.grid(True)
            plt.xticks(np.arange(len(reports))*step)
            plt.savefig(join(plot_dir, '{}_{}.png'.format(prefix, m)))
            plt.close()
            plog += 'Figure saved: {}_{}.png\n'.format(prefix, m)
        return plog

    def plot_final_evas(self, report, plot_dir, prefix):
        pass

    def save(self, reports, folder, prefix):
        np.save(join(folder, '{}.npy'.format(prefix)), reports)
        plog = 'All reports saved to {}'.format(
            join(folder, '{}.npy'.format(prefix))
        )
        return plog


class MultiModalityMetaSREvaluation(MetaSREvaluation):

    def __init__(self, modalities, metrics, sr_factors, gpu_id=-1, record_mode='full'):
        self.modalities = modalities
        super(MultiModalityMetaSREvaluation, self).__init__(
            metrics, sr_factors, gpu_id, record_mode
        )

    def __call__(self, rec_imgs, samples):
        if isinstance(samples, dict):
            samples = [samples]
            rec_imgs = [rec_imgs]
        assert len(rec_imgs) == len(samples), 'Numbers of rec_imgs and samples should match'

        all_reports = {}
        for i in range(len(self.modalities)):
            modality = self.modalities[i]
            report = {}
            for s in self.sr_factors:
                gt_imgs = [sample[s]['gt'][:, :, i:i+1] for sample in samples]
                rec_imgs_with_scale = [rec_img[s][:, :, i:i+1] for rec_img in rec_imgs]
                report_with_scale = self.func(gt_imgs, rec_imgs_with_scale, int(np.ceil(s)))
                for m in report_with_scale:
                    report['{}_{}'.format(m, s)] = report_with_scale[m]

            all_reports[modality] = report
        return all_reports

    def print(self, report):
        plog = ''
        for m in report:
            plog += '\n{} performance:\n'.format(m)
            plog += super(MultiModalityMetaSREvaluation, self).print(report[m])

        return plog

    def plot_process(self, reports, plot_dir, prefix='', step=1):
        plog = ''
        for m in self.modalities:
            m_reports = [r[m] for r in reports]
            plog += '\nPlotting {}\n'.format(m)
            plog += super(MultiModalityMetaSREvaluation, self).plot_process(
                m_reports, plot_dir, '{}{}'.format(prefix, m), step
            )
        return plog

    def save(self, reports, folder, prefix):
        plog = ''
        for m in self.modalities:
            m_reports = [r[m] for r in reports]
            plog += '\n{}\n'.format(m)
            plog += super(MultiModalityMetaSREvaluation, self).save(
                m_reports, folder, '{}{}'.format(prefix, m)
            )
        return plog

    def stack_eva_reports(self, reports):
        stacked_report = {}
        for m in self.modalities:
            m_reports = []
            for r in reports:
                m_reports.append(r[m])

            stacked_report[m] = super(MultiModalityMetaSREvaluation, self).stack_eva_reports(m_reports)
        return stacked_report
