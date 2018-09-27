# enas_nni
This code is for running enas code on nni system.  
link:https://github.com/Microsoft/nni  
link:https://github.com/melodyguan/enas  

Install:  
  1 Need to install nni system first.  
  link:https://github.com/Microsoft/nni You need to choose the dev-enas branch.  
  Use command:   
    pip3 install -v --user git+https://github.com/Microsoft/nni.git@dev-enas  
    source ~/.bashrc  
  2 Modify the codeDir of  enas_nni/nni/examples/trials/enas/config.yml.  
  3 Download the cifar10 data at https://www.cs.toronto.edu/~kriz/cifar.html (python version). You need to put data at  enas_nni/nni/examples/trials/enas/data and rename the cifar-10-batches-py as cifar10 .   
  4 Run the command: nnictl create --config ~/chicm/nni/examples/trials/enas/config.yml (It depends on where you put the config on)  

  
