% PCA Thresholding, and using PCA for classifiers.
% By Patrick Savoie and Alex Odonal

clear
clc


tic
% Load cropped images
a = dir('50k\*.jpg')

featvecmat = zeros(size(a,1),4096);             
for i = 1 : size(a,1)       
%    I = imread(['celeba_low/',a(i).name]);
   I = rgb2gray(imread(['50k\',a(i).name])); % Convert cropped images to rgb
   featvecmat(i,:) = I(:);       % Transform image into feature vector. Do this for each image
end
z = 1

featvecmat = double(featvecmat);            %Convert features to doubles so it can be used in MATLAB's pca function
[coeff, score, latent] = pca(featvecmat);
z = 2

var_captured = 0;           % Variance captured from features (between 0 and 1)
i = 1;
while (i<size(coeff,2) && var_captured<0.98) %Loop through until 98% of variance is captured by eigen vectors
    var_captured = sum(latent(1:i))/sum(latent);               % Proportion of variance up to the ith eigen vector of pca 
    i = i + 1;
end
num_components = i;



%% Show put all eigenfaces in vectors


% im = cell(1,size(coeff,2));
im = cell(1,100)
Ibasis = uint8(zeros(64,64));
coenew = zeros(1,4096);
for z = 1 : 100
    m = 1;
    % Unfold feature vector into image
    coenew = coeff(:,z) - min((coeff(:,z))); 
    coenew = uint8(255*coenew/max(coenew));

  % Different type of image normalization
%   coenew(1:12288/3) = coeff(1:12288/3,z) - min(coeff(1:12288/3,z));
%   coenew(1+12288/3:2*12288/3,z) = coeff(1+12288/3:2*12288/3,z) - min(coeff(1+12288/3:2*12288/3,z));
%   coenew(1+2*12288/3:12288) = coeff(1+2*12288/3:12288,z) - min(coeff(1+2*12288/3:12288,z));
%   
%   coenew(1:12288/3) = 255*coenew(1:12288/3)/max(coenew(1:12288/3));
%   coenew(1+12288/3:2*12288/3) = 255*coenew(1+12288/3:2*12288/3)/max(coenew(1+12288/3:2*12288/3));
%   coenew(1+2*12288/3:12288) = 255*coenew(1+2*12288/3:12288)/max(coenew(1+2*12288/3:12288));


for j = 1:64
    for k = 1:64
        Ibasis(k,j) = uint8(coenew(m));  % Unfolded basis image
        m = m + 1;
    end
end
im{z} = Ibasis;
disp(z);
end


mean_face = uint8(mean(featvecmat))
m = 1;
for j = 1:64
    for k = 1:64
        im_mean_sunset(k,j) = uint8(mean_face(m));  % Unfolded basis image
        m = m + 1;
    end
end


% Show faces

imagearr = cat(4,im_mean_sunset,im{1:99});
montage(imagearr, 'DisplayRange',[0,255]);


%% Determining the maximum tolerable reconstruction error with a finite number of pca components. Finding the threshold with 50000 images to get an idea of the 
% Threshold, due to very long run times with more images
    % For the Oth image
    mean_face = double(mean(featvecmat));
for O = 1:size(featvecmat,1)

    imvecO = featvecmat(O,:)'- mean_face';
    proj_vecO = coeff(:,1:num_components)'*imvecO;            % Project the i first principle components    
    imavec_reconstructed = double(mean_face');
    for mmm = 1:num_components
        imavec_reconstructed = imavec_reconstructed + proj_vecO(mmm)*coeff(:,mmm);
    end
    
    
    error_reconstruct1(O) = sum((featvecmat(O,:)'-imavec_reconstructed).^2);
    
    
    
    if(mod(O,50)==0)
       disp(O) 
    end
end

error_threshold = max(error_reconstruct1)


%% Load generated faces and convert to grayscale

a = dir('generated_faces/generated_faces/*.png');
clear error_reconstruct;
clear errs_sorted;


featvecmat2 = zeros(size(a,1),4096);
for i = 1 : size(a,1)       
   I = rgb2gray(imread(['generated_faces/generated_faces/',a(i).name]));
   featvecmat2(i,:) = I(:);       % Transform generated images into feature matrix
end
featvecmat2 = double(featvecmat2);


%% Determining if faces are recognizable or not
newfeatvec = zeros(size(a,1),num_components);              %New feature vector array of generated images 
mean_face = double(mean(featvecmat2));
for O = 1:size(a,1)
    
    % For the Oth image
    imvecO = featvecmat2(O,:)'- (mean_face');
    proj_vecO = coeff(:,1:num_components)'*imvecO;            % Project the i first principle components
    
    imavec_reconstructed = double(mean_face');
    for mmm = 1:num_components
        imavec_reconstructed = imavec_reconstructed + proj_vecO(mmm)*coeff(:,mmm);
    end
    
    
    error_reconstruct(O) = sum((featvecmat2(O,:)'-imavec_reconstructed).^2);
    
    

%
    if error_reconstruct(O)< error_threshold
        face(O) = 1; % Image is face if it was reconstructed with low error from the n principle components
    else
        face(O) = 0;
    end
    
    if(mod(O,50)==0)
       disp(O) 
    end
end



% Evaluate accuracy
errs_sorted = sort(error_reconstruct)
errmax = max(error_reconstruct)
a2 = find(error_reconstruct==errs_sorted(5120))
acc_pca = sum(face)/size(face,2)


%% disaplying "good", and "bad" faces based on reconstruction error of pca

bad_gen = cell(1,5120-sum(face)+1);  % Bad generated images

index = 1; 
for i = 5120:-1:sum(face)           % Quickly showing images that don't work   
   bad_gen{index} = imread(['generated_faces/generated_faces/',a(find(error_reconstruct==errs_sorted(i))).name]);  % Name of images with highest reconstruction error
   index = index + 1;
end

figure(2)
bad_gen_arr = cat(4,bad_gen{1:5120-sum(face)+1});
montage(bad_gen_arr, 'DisplayRange',[0,255]);


index = 1; 
for i = 1:64          % Quickly showing images that don't work   
   good_gen{index} = imread(['generated_faces/generated_faces/',a(find(error_reconstruct==errs_sorted(i))).name]);  % Name of images with highest reconstruction error
   index = index + 1;
end

figure(3)
good_gen_arr = cat(4,good_gen{1:64});
montage(good_gen_arr, 'DisplayRange',[0,255]);

index = 1; 
for i = sum(face)-65: sum(face)-1         % Quickly showing images that just barely failed  
   good_gen{index} = imread(['generated_faces/generated_faces/',a(find(error_reconstruct==errs_sorted(i))).name]);  % Name of images with highest reconstruction error
   index = index + 1;
end

figure(4)
good_gen_arr = cat(4,good_gen{1:64});
montage(good_gen_arr, 'DisplayRange',[0,255]);




toc
%% Classifiers
tic
a  = dir('non_face\Non-face\*.jpg');
featvecmat_all = zeros(size(featvecmat,1)+size(a,1),4096);
featvecmat_all(1:size(featvecmat,1),:) = featvecmat;

% Labeling images
classes = zeros(size(featvecmat,1)+size(a,1),1);
classes(1:size(featvecmat,1)) = 1; % First images are images of faces (Represented by 1), others are faces of non-faces (Represented by 0)

% non_faces = cell(1,50000);  % Non-face images

% m=1;
% ind = 1;
ind = 1;
for i = size(featvecmat,1)+1 : size(featvecmat,1)+size(a,1)-1
%     if ind== 5992
%         ind = 23992;
%     end
%     I = rgb2gray(reshape(data(ind,:),32,32,3)); % Convert cropped images to rgb
%     I = imresize(I,[64,64]);         % Resizing the image to a 64 by 64 image
%     I = imrotate(I,-90);             % Rotate images so that it's facing the correct orientation
%     imwrite(I,['Non-face/',num2str(i),'.jpg']);
    I = imread(['non_face\Non-face\',a(ind).name]);
    featvecmat_all(i,:) = I(:);
%     non_faces{m} = I;
%     featvecmat_all(i,:) = I(:);       % Transform image into feature vector. Do this for each image
    if mod(i,1000) == 0
        disp(i)
    end
    ind  = ind + 1;
%     m=m+1;
%     ind = ind + 1;
end
% figure(5)
% non_faces_arr = cat(4,non_faces{8200:8300});
% montage(non_faces_arr, 'DisplayRange',[0,255]);

featvecmat = double(featvecmat);            %Convert features to doubles so it can be used in MATLAB's pca function
toc
%%
% Perform pca on all face and non-face images.
[coeff, score, latent] = pca(featvecmat);

var_captured = 0;           % Variance captured from features (between 0 and 1)
i = 1;
while (i<size(coeff,2) && var_captured<0.98) %Loop through until 98% of variance is captured by eigen vectors
    var_captured = sum(latent(1:i))/sum(latent);               % Proportion of variance up to the ith eigen vector of pca 
    i = i + 1;
end
max_num_components = i;

% Using 10 fold cross validation
partition = cvpartition(classes,'KFold',10);

%%
% Project images on new feature space (First n principle components that
% represent most of the variance
    mean_face = double(mean(featvecmat_all));
    newfeatvec_all = zeros(size(featvecmat_all,1),max_num_components);
for O = 1:size(featvecmat_all,1)
    imvecO = featvecmat_all(O,:)'- (mean_face');
    newfeatvec_all(O,:) = coeff(:,1:max_num_components)'*imvecO;            % Project the i first principle components
    if (mod(O,500) == 0)
        disp(O)
    end
end

% Different number of features used
eigen_epochs = floor(max_num_components/10):floor(max_num_components/10):max_num_components;




%%
% Project generated images on new feature space (First n principle components that
% represent most of the variance
    mean_face = double(mean(featvecmat_all));
    newfeatvecmat2 = zeros(size(featvecmat2,1),max_num_components);
for O = 1:size(featvecmat2,1)
    imvecO = featvecmat2(O,:)'- (mean_face');
    newfeatvecmat2(O,:) = coeff(:,1:max_num_components)'*imvecO;            % Project the i first principle components
    if (mod(O,500) == 0)
        disp(O)
    end
end

%%
% LDA
% for each feature, find the accuracy
ind = 1;
batchval = [1,10,40,eigen_epochs];
tic
for q = batchval
    for L = 1:10 % for each partition

        % separate training and test data
        idxTrn = training(partition,L); % training set indices
        idxTest = test(partition,L);    % test set indices

        % train discriminator for each feature
        linDisc = fitcdiscr(newfeatvec_all(idxTrn,1:q),classes(idxTrn));

        % n is the numer of test elements 
        n = size(classes(idxTest),1);

        %Accuracy set q of cross validation
        accuracy(ind,L) = sum(classes(idxTest)==predict(linDisc, newfeatvec_all(idxTest,1:q)))/n;

    end
        accuracyMean(ind) = mean(accuracy(ind,:)');
        accuracystd(ind) = std(accuracy(ind,:)')
        ind = ind + 1
end
q = batchval(find(accuracyMean == max(accuracyMean)));
toc

 
%% Classify images with optimal parameters with LDA with PCA component
a = dir('generated_faces/generated_faces/*.png');
linDisc = fitcdiscr(newfeatvec_all(idxTrn,1:q),classes(idxTrn));
values_no = predict(linDisc, newfeatvecmat2(:,1:q));
acc = sum(values_no)/size(newfeatvecmat2,1);

m = find(values_no == 0)

in = 1; 
for i = m(1:64)           % Quickly showing 64 of the images that didn't work  
   bad_gen{index} = imread(['generated_faces/generated_faces/',a(m).name;  % Name of images with highest reconstruction error
   in = in + 1;
end

figure(7)
bad_gen_arr = cat(4,bad_gen{1:64});
montage(bad_gen_arr, 'DisplayRange',[0,255]);

%% KNN
ind = 1;
tic
L = 1;
% Test with different values of k
for k = 1:10
%     arithmetic_index = 1;
    for q = max(eigen_epochs)
            idxTrn = training(partition,L); % training set indices
            idxTest = test(partition,L);    % test set indices


            KNNDiscsex = fitcknn(newfeatvec_all(idxTrn,1:q),classes(idxTrn), 'NumNeighbors',k);        % Update value of k with each iteration

            % n is the numer of test elements 
            n = size(classes(idxTest),1);

            %Accuracy set q of cross validation
%             accuracy_k(arithmetic_index) = sum(classes(idxTest)==predict(KNNDiscsex, newfeatvec_all(idxTest,1:q)))/n;
%             arithmetic_index = arithmetic_index + 1

    end
    accuracyMean_k(k) = sum(classes(idxTest)==predict(KNNDiscsex, newfeatvec_all(idxTest,1:q)))/n;
    disp(k)
end
toc
% Choose the value of k that yields the max accuracy
k = find(accuracyMean_k ==  max(accuracyMean_k));


%%
% Test knn with different distance metrics
distance_metric = cell(1,4);
distance_metric = {'cityblock','chebychev','euclidean',	'mahalanobis'}

for m = 1:4
    for q = max(eigen_epochs)
            idxTrn = training(partition,L); % training set indices
            idxTest = test(partition,L);    % test set indices

            KNNDiscsex = fitcknn(newfeatvec_all(idxTrn,1:q),classes(idxTrn), 'NumNeighbors',k, 'Distance', distance_metric{m} );        % Update value of distance metric with each iteration
            % n is the numer of test elements 
    %         n = size(classes(idxTest),1);


            accuracyKNNsex(arithmatic_index) = sum((predict(KNNDiscsex, newfeatvec_all(:,1:q))) == classes(idxTest))/n;
            arithmatic_index = arithmatic_index + 1;
            
            % n is the numer of test elements 
            n = size(classes(idxTest),1);

    end
    accuracyMean_k_dist(m) = sum(classes(idxTest)==predict(KNNDiscsex, newfeatvec_all(idxTest,1:q)))/n;
    disp(m)
end


%%
m = accuracyMean_k_dist==max(accuracyMean_k_dist);          % Select distance metric that maximizes accuracy

for q = eigen_epochs
    for L = 1:5                     % Due to extremely long run time, perform 5 fold cross validation
    idxTrn = training(partition,L); % training set indices
    idxTest = test(partition,L);    % test set indices
    
    KNNDiscsex = fitcknn(newfeatvec_all(idxTrn,1:q),classes(idxTrn), 'NumNeighbors',k);        % Update value of distance metric with each iteration
    % n is the numer of test elements
    %         n = size(classes(idxTest),1);
    
    
    accuracyKNNsex(arithmatic_index) = sum((predict(KNNDiscsex, newfeatvec_all(:,1:q))) == classes(idxTest))/n;
    arithmatic_index = arithmatic_index + 1;
    
    % n is the numer of test elements
    n = size(classes(idxTest),1);
    end
end
    accuracyMean_k_dist(m) = sum(classes(idxTest)==predict(KNNDiscsex, newfeatvec_all(idxTest,1:q)))/n;
    disp(m)




% Classify generated images with optimal parameter
%% SVM
arithmatic_index = 1;
for ep = eigen_epochs
    for L = 1:1
    SVMDiscrace =  fitcsvm(x(:,1:ep),y_race); % fitecoc is used for multiple classes
    accuracySVMrace(arithmatic_index) = sum((predict(SVMDiscrace, x_test(:,1:ep))) == y_race_test')/2000;
    arithmatic_index = arithmatic_index + 1;
    end
end

