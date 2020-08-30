import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.evaluator import Evaluator

def eval(opt, dataset, model, evaluator):
    model.eval()
    evaluator.reset()
    eval_start_time = time.time()
    for i, data in enumerate(dataset):
        model.set_input(data)
        preds = model.test()
        evaluator.update(preds)
    eval_time = time.time() - eval_start_time
    res = '==>eval time: {:.0f},'.format(eval_time)
    metric, select_score = evaluator.summary(eval_mode = 'edge_rel') # edge_rel | lloc
    res += metric
    return res, select_score

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    torch.manual_seed(10)
    if len(opt.gpu_ids) > 0:
        torch.cuda.manual_seed(10)
    # train dataset
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_size = len(train_dataset)    # get the number of images in the dataset.
    print('The number of training images = %d. Trainset: %s' % (train_size, opt.dataroot))

    opt.print_freq = train_size//10 # print 10 times for each epoch
    opt.save_latest_freq = train_size//opt.batch_size*opt.batch_size # save latest model and evaluate the performance after every epoch

    # test dataset
    opt.phase    = 'test'
    test_dataset = create_dataset(opt)
    evaluator = Evaluator(opt)
    test_size    = len(test_dataset)
    print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))
    opt.phase = 'train'

    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)  
    total_iters = 0                
    best_res    = 0.0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): 
        epoch_start_time = time.time()  
        iter_data_time = time.time()    
        epoch_iter = 0                  

        for i, data in enumerate(train_dataset): 
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)        
            model.optimize_parameters() 

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                print_res, avg_res = eval(opt, test_dataset, model, evaluator) 
                visualizer.print_current_val(epoch, epoch_iter, print_res)
                if avg_res > best_res:
                    best_res = avg_res
                    print('saving the best model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    model.save_networks('best')
                    #model.metric = best_acc
                print_best = 'current avg acc: {:.4f}, best acc: {:.6f}'.format(avg_res, best_res)
                visualizer.print_current_val(epoch, epoch_iter, print_best)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()     