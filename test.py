import os, time
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from tqdm import tqdm
from util.evaluator import Evaluator

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 1   
    opt.batch_size = 1    
    opt.serial_batches = True 
    test_dataset = create_dataset(opt)
    test_size    = len(test_dataset)
    print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))
    opt.num_test = test_size
    evaluator = Evaluator(opt)

    model = create_model(opt)      
    model.setup(opt)               
    
    #save_dir = os.path.join(os.getcwd(), opt.results_dir, opt.name, opt.dataroot.split('/')[-1], '%s_%s' % (opt.phase, opt.epoch)) 
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    model.eval()
    evaluator.reset()
    eval_start_time = time.time()
    for data in tqdm(test_dataset):
        model.set_input(data)
        preds = model.test()
        evaluator.update(preds)
    eval_time = time.time() - eval_start_time
    res = '==>Evaluation time: {:.0f},\n'.format(eval_time)
    metric, select_score = evaluator.summary(eval_mode = 'edge_rel | lloc')
    res += metric
    print(res)