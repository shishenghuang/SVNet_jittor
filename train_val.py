
import numpy as np 
import jittor as jt 
import os 
import tqdm 
from network import *
from dataset import * 
from loss import *


if __name__ = '__main__':

    freeze_random_seed()

    parser = argparse.ArgumentParser(decription='SVNet')
    parser.add_argument('--train_file' , type=str , default='./data/train.h5', metavar='N', help='The train file data')
    parser.add_argument('--val_file', type=str, default='./data/val.h5' , help='Evaluate file')
    parser.add_argument('--batch_size' , type=int , default=32, metavar='batch_size', help = 'Size of batch')
    parser.add_argument('--lr' , tpye=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--num_points', type=int, default=1024,help='Points Number')
    parser.add_argument('--num_k', type=int, default=8, help='Number Neighbors')
    parser.add_argument('--num_class', type=int, default=21, help='Number Classes')
    parser.add_argument('--epoches' , type=int, default=10, help="Train Epoches")

    args = parser.parse_args()

    train_files = args.train_file
    val_files = args.val_file
    num_points = args.num_points
    num_k = args.num_k
    num_class = args.num_class

    lr = args.lr 
    epoches = args.epoches

    net = SVNet(num_points_ = num_points, num_k_ = num_k , num_class_ = num_class)

    optimizer = jt.nn.Adam(net.parameters(), lr = lr )

    train_dataloader = SVNetTrainDataSet(train_files, num_points , num_k , num_class)
    val_dataloader =SVNetTestDataSet(val_files , num_points , num_k , num_class)

    for epoch in range(epoches):

        net.train()

        if epoch % 10 == 0:
            jt.save(net.state_dict(), 'checkpoints/models/model_%d.th' % (epoch))

        for idx , (pnts_cuda, labels_cuda) in enumerate(train_dataloader)

            points = pnts_cuda[:,:3].cuda()
            features = pnts_cuda[:,3:].cuda()

            target_label = labels_cuda.cuda()

            pred_label = net(points , features)

            loss = weighted_cross_entropy(pred_label , target_label)
            optimizer.zero_grad()
            #loss.backward()
            optimizer.step(loss)

            print("SVNet Train -- INFO -- epoch : %d, idx : %d, loss : %f\n" % (epoch , idx , loss))
        




