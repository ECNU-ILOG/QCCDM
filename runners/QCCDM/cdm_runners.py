from method.QCCDM.qccdm import QCCDM
from method.Baselines.NCDM.ncdm_qccdm import NCDM as NCDMQ
from method.Baselines.NCDM.ncdm import NCDM
from method.Baselines.KANCD.kancd import KaNCD
from method.Baselines.KANCD.kancd_qccdm import KaNCD as KANCDQ
from method.Baselines.DINA.dina import DINA
from method.Baselines.DINA.dina_qccdm import DINA as DINAQ
from method.Baselines.MIRT.mirt import MIRT
from method.Baselines.HIERCDF.hiercdf import HierCDM as HIERCDF
from method.Baselines.HIERCDF.hiercdf_qccdm import HierCDM as HIERCDFQ
from method.Baselines.KSCD.kscd import KSCD
from method.Baselines.RCD.rcd import RCD
from runners.QCCDM.utils import save


def qccdm_runner(config: dict):
    """
    choose the mode of QCCDM
    '1' only SCM '2' only Q-augmented '12' both
    """
    if config['mode'] == '1':
        qccdm_causal = QCCDM(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                             graph=config['graph'],
                             q_matrix=config['q'], device=config['device'], mode=config['mode'],
                             dtype=config['dtype'], num_layers=config['num_layers'], nonlinear=config['nonlinear'])
        qccdm_causal.train(config['np_train'], config['np_test'], epoch=config['epoch'],
                           batch_size=config['batch_size'], q=config['q'])
        save(config, qccdm_causal.mas_list)
    elif config['mode'] == '2':
        qccdm_augmented = QCCDM(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                                graph=config['graph'],
                                q_matrix=config['q'], device=config['device'], mode=config['mode'],
                                dtype=config['dtype'], lambda_reg=config['lambda'], num_layers=config['num_layers'], q_aug=config['q_aug'])
        qccdm_augmented.train(config['np_train'], config['np_test'], epoch=config['epoch'],
                              batch_size=config['batch_size'], q=config['q'])
        save(config, qccdm_augmented.mas_list)

    else:
        qccdm = QCCDM(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                      graph=config['graph'],
                      q_matrix=config['q'], device=config['device'], mode=config['mode'],
                      dtype=config['dtype'], lambda_reg=config['lambda'], num_layers=config['num_layers'],
                      nonlinear=config['nonlinear'], q_aug=config['q_aug'])
        qccdm.train(config['np_train'], config['np_test'], epoch=config['epoch'], batch_size=config['batch_size'],
                    q=config['q'])
        save(config, qccdm.mas_list)


def ncdmq_runner(config: dict):
    ncdmq = NCDMQ(student_n=config['stu_num'], exer_n=config['prob_num'], knowledge_n=config['know_num'],
                  Q_matrix=config['q'], lambda_reg=config['lambda'], device=config['device'])
    ncdmq.train(np_train=config['np_train'], np_test=config['np_test'], q=config['q'], batch_size=config['batch_size'],
                epoch=config['epoch'])


def ncdm_runner(config: dict):
    ncdm = NCDM(student_n=config['stu_num'], exer_n=config['prob_num'], knowledge_n=config['know_num'],
                device=config['device'])
    ncdm.train(np_train=config['np_train'], np_test=config['np_test'], q=config['q'], batch_size=config['batch_size'],
               epoch=config['epoch'])
    save(config, ncdm.mas_list)


def kancdq_runner(config: dict):
    kancdq = KANCDQ(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                    Q_matrix=config['q'], lambda_reg=config['lambda'], device=config['device'], dim=20)
    kancdq.train(np_train=config['np_train'], np_test=config['np_test'], q=config['q'], batch_size=config['batch_size'],
                 epoch=config['epoch'])


def kancd_runner(config: dict):
    kancd = KaNCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'],
                  device=config['device'], dim=20)
    kancd.train(np_train=config['np_train'], np_test=config['np_test'], q=config['q'], batch_size=config['batch_size'],
                epoch=config['epoch'])
    save(config, kancd.mas_list )


def dina_runner(config: dict):
    dina = DINA(user_num=config['stu_num'], item_num=config['prob_num'], hidden_dim=config['know_num'],
                device=config['device'])
    dina.train(np_train=config['np_train'], np_test=config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])


def dinaq_runner(config: dict):
    dinaq = DINAQ(user_num=config['stu_num'], item_num=config['prob_num'], hidden_dim=config['know_num'],
                  device=config['device'], Q_matrix=config['q'])
    dinaq.train(np_train=config['np_train'], np_test=config['np_test'], epoch=config['epoch'], q=config['q'],
                batch_size=config['batch_size'])


def mirt_runner(config: dict):
    mirt = MIRT(user_num=config['stu_num'], item_num=config['prob_num'], latent_dim=16, device=config['device'])
    mirt.train(config['np_train'], config['np_test'], epoch=config['epoch'], q=config['q'],
               batch_size=config['batch_size'])


def hiercdf_runner(config: dict):
    hiercdf = HIERCDF(n_user=config['stu_num'], n_item=config['prob_num'], n_know=config['know_num'], hidden_dim=512,
                      know_graph=config['hier'], device=config['device'])
    hiercdf.train(config['np_train'], config['np_test'], epoch=config['epoch'], q=config['q'],
                  batch_size=config['batch_size'])
    save(config, hiercdf.mas_list)


def hiercdfq_runner(config: dict):
    hiercdfq = HIERCDFQ(n_user=config['stu_num'], n_item=config['prob_num'], n_know=config['know_num'], hidden_dim=512,
                        know_graph=config['hier'], device=config['device'], Q_matrix=config['q'])
    hiercdfq.train(config['np_train'], config['np_test'], epoch=config['epoch'], q=config['q'],
                   batch_size=config['batch_size'])


def kscd_runner(config):
    kscd = KSCD(stu_num=config['stu_num'], prob_num=config['prob_num'], know_num=config['know_num'], dim=20,
                device=config['device'])
    kscd.train(config['np_train'], config['np_test'], q=config['q'], batch_size=config['batch_size'],
               epoch=config['epoch'])
    save(config, kscd.mas_list)


def rcd_runner(config):
    rcd = RCD(config=config)
    rcd.train()
    save(config, rcd.mas_list)


def get_runner(method: str):
    if 'qccdm' in method:
        return qccdm_runner
    elif method == 'ncdmq':
        return ncdmq_runner
    elif method == 'ncdm':
        return ncdm_runner
    elif method == 'kancdq':
        return kancdq_runner
    elif method == 'kancd':
        return kancd_runner
    elif method == 'mirt':
        return mirt_runner
    elif method == 'dinaq':
        return dinaq_runner
    elif method == 'dina':
        return dina_runner
    elif method == 'hiercdf':
        return hiercdf_runner
    elif method == 'hiercdfq':
        return hiercdfq_runner
    elif method == 'kscd':
        return kscd_runner
    elif method == 'rcd':
        return rcd_runner
    else:
        raise ValueError('This method is currently not supported.')
