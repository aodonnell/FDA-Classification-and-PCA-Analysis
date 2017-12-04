%% FDA magic it is recomended that your machine has atleast 8GB of ram. 4GB will most likely not be enough
% script made by alex odonnell and patrick savoie 
fprintf(['The program has been paused. This flag has been placed here to avoid \n '...
    'the scenario where the user accidentally runs the entire script. It is \n'...
    'recommended that you run it by section in order to gain a better understanding \n'...
    'of the methods used during the experiment. Press any key to continue\n'])
pause;

%% start by clearing the workspace
% =========================================================================
fprintf(['Are you sure you wish to clear the workspace? Press any key to continue\n'])
pause;
clear
clc
close all

%% setting parameters
% =========================================================================
facefolder = 'img_align_celeba_crop/img_align_celeba_crop/';
nonfacefolder = 'Non-face/';
generated = 'generated_faces/generated_faces/';

a = dir(facefolder);
a = a(3:end);
b = dir(nonfacefolder);
b = b(3:end);
c = dir(generated);
c = c(3:end);

n_imgs = 48000; % total number of images for each class
imgsize = 64*64*1; % 64x64 b and w

% initialize features
X = zeros(2*(n_imgs),imgsize);
Xgen = zeros(5000,imgsize);
Zgen = zeros(1,5000);

% set up classes
Y = zeros(2*n_imgs,1);
Y(1:n_imgs) = 1;

% Using 10 fold cross validation
partition = cvpartition(Y,'KFold',10);

%% Read images and create feature matrix
% =========================================================================
tic
for i = 1 : n_imgs
    if mod(i,1000) == 0
        disp(i);
    end
    I = rgb2gray(imread([facefolder,a(i).name]));
    X(i,:) = I(:);
    I = imread([nonfacefolder,b(i).name]);
    X((i+n_imgs),:) = I(:);
 
end
disp('Done reading images.')
toc

%% clear unused variables
clear a b facefolder nonfacefolder

%% run FDA. May take a while. Load W and Z from the model to avoid this.
% =========================================================================
tic
[Z,W] = FDA(X',Y);
disp('Done FDA. New dimentionality reduced feature is stored in Z.')
toc

%% test FDA by transforming some artificial faces
% =========================================================================
tic
for i = 1:5000
    if mod(i,1000) == 0
        disp(i);
    end
    I = rgb2gray(imread([generated,c(i).name]));
    Xgen((i),:) = I(:);
    Zgen(i) = W'*double(Xgen((i),:))';
end
disp('Done transforming artificaial images.')
toc

%% clear some more unused variables
clear c generated

%% histogram of our new feature. Zgen is used here to show the relationship
% between the distributions of real and generated faces.
% =========================================================================
figure(1)
hold on;
histogram(Z(Y==0),'FaceColor',[0.8 0 0])
histogram(Z(Y==1),'FaceColor',[0 0.8 0])
histogram(Zgen,'FaceColor',[0 0 0.8])
hold off;

%%
% LDA
% =========================================================================
LDApredictions = zeros(10,5000);

tic
for L = 1:10 % for each partition

    % separate training and test data
    idxTrn = training(partition,L); % training set indices
    idxTest = test(partition,L);    % test set indices

    % train discriminator for each feature
    linDisc = fitcdiscr(Z(idxTrn)',Y(idxTrn));

    % n is the numer of test elements 
    n = size(Y(idxTest),1);

    %Accuracy set q of cross validation
    accuracy(L) = sum(Y(idxTest) == predict(linDisc, Z(idxTest)'))/n;
    LDApredictions(L,:) = predict(linDisc, Zgen')/n;

end
disp('LDA complete.')
toc

%% visualize a subset of the generated images that failed the linear discriminator

failidx = find(LDApredictions(1,:) == 0);


for i = 1:64
    failedimg{i} = reshape(Xgen(failidx(i),:),[64,64]);
end
imagemat = cat(4,failedimg{1:64});

figure(2)
montage(imagemat, 'DisplayRange', [0 255]);

%% clear more values
clear failidx imagemat

%% Select optimum K value and distance metric
% =========================================================================
arithmetic_index = 1;
ind = 1;

max_k = 30;

distance_metric = {'cityblock','chebychev','euclidean'};

accuracyKNN = zeros(10,1);
accuracyMean_k = zeros(30,1);
accuracyMean_m = zeros(3,1);
k_best = accuracyMean_m;
% Test with different values of k

for m = 1:3
    tic
    for k = 1:max_k
        for L = 1:10 % for each partition
            idxTrn = training(partition,L); % training set indices
            idxTest = test(partition,L);    % test set indices

            KNNDisc = fitcknn(Z(idxTrn)',Y(idxTrn), 'NumNeighbors', k, 'Distance', distance_metric{m} );        % Update value of k with each iteration
            % n is the numer of test elements 
            % 
            n = size(Y(idxTest),1);
            
            accuracyKNN(L) = sum((predict(KNNDisc, Z(idxTest)')) == Y(idxTest))/n;

            % n is the numer of test elements 
            n = size(Y(idxTest),1);

        end  
    accuracyMean_k(k)  = mean(accuracyKNN);
    end
    accuracyMean_m(m) =  mean(accuracyMean_k);
    disp(['KNN complete using ' distance_metric{m} ' distance.'])
    % Choose the value of k that yields the best perfo rmance
    k_best(m) = find(accuracyMean_k ==  max(accuracyMean_k));
    disp(['Optimal k value below ' num2str(max_k) ' for ' distance_metric{m} 'distance: ' num2str(k_best(m)) '.'] )
    toc
end

%% Free up some more space
clear accuracyMean_m accuracyMean_k

%% train KNN using best model
% this is also after we concluded that the distance metrics all work the
% same so we'll just use Euclidean distance (m = 1). Retraining KNN is a
% sideffect of clearing our model everytime. We didn't do this in order to
% preserve as many resources as we can
m = 1;
% =========================================================================
KNNpredictions = zeros(10,5000);
k = k_best(m);
tic
for L = 1:10 % for each partition
    idxTrn = training(partition,L); % training set indices
    idxTest = test(partition,L);    % test set indices

    KNNDisc = fitcknn(Z(idxTrn)',Y(idxTrn), 'NumNeighbors', k, 'Distance', distance_metric{m} );        % Update value of k with each iteration
    % n is the numer of test elements 
    % 
    n = size(Y(idxTest),1);

    accuracyKNN(L) = sum((predict(KNNDisc, Z(idxTest)')) == Y(idxTest))/n;

    % n is the numer of test elements 
    n = size(Y(idxTest),1);
    
    % test with artificial faces
    KNNpredictions(L,:) = predict(KNNDisc, Zgen')/n;
end
accuracyMean = mean(accuracyKNN);
disp(['KNN complete. With optimal parameters, mean accuracy for all partitions: ' num2str(accuracyMean) '.'])
toc


%% visualize a subset of the generated images that failed the KNN discriminator
failidx = find(KNNpredictions(1,:) == 0);

for i = 1:64
    failedimg{i} = reshape(Xgen(failidx(i),:),[64,64]);
end
imagemat = cat(4,failedimg{1:64});

figure(3)
montage(imagemat, 'DisplayRange', [0 255]);


%% SVM (takes roughly 15 mins for an i7 CPU with 16 gigs of RAM)
% =========================================================================
SVMpredictions = zeros(10,5000);

tic
for L = 1:10 % for each partition
    
    
    % separate training and test data
    idxTrn = training(partition,L); % training set indices
    idxTest = test(partition,L);    % test set indices

    % train discriminator for each feature
    SVMDisc = fitcsvm(Z(idxTrn)',Y(idxTrn));

    % n is the numer of test elements 
    n = size(Y(idxTest),1);

    %Accuracy set q of cross validation
    accuracy(L) = sum(Y(idxTest) == predict(SVMDisc, Z(idxTest)'))/n;
    SVMpredictions(L,:) = predict(SVMDisc, Zgen')/n;

end
disp('SVM complete.')
toc

%% visualize a subset of the generated images that failed the SVM discriminator
failidx = find(SVMpredictions(1,:) == 0);

for i = 1:64
    failedimg{i} = reshape(Xgen(failidx(i),:),[64,64]);
end
imagemat = cat(4,failedimg{1:64});

figure(4)
montage(imagemat, 'DisplayRange', [0 255]);

