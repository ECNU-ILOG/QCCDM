from method.QCCDM.qccdm import QCCDM
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



def get_runner(method: str):
    if 'qccdm' in method:
        return qccdm_runner
    else:
        raise ValueError('This method is currently not supported.')
