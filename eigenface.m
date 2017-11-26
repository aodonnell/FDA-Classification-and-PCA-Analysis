clear
clc


% Load cropped images
a = dir('img_align_celeba_crop/*.jpg')

featvecmat = zeros(size(a,1),4096);             
for i = 1 : size(a,1)       
%    I = imread(['celeba_low/',a(i).name]);
   I = rgb2gray(imread(['img_align_celeba_crop/',a(i).name])); % Convert cropped images to rgb
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
Ibasis = uint8(zeros(64,64,3));
coenew = zeros(1,12288);
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


    for i = 1:3
        for j = 1:64
            for k = 1:64
                Ibasis(k,j,i) = uint8(coenew(m));  % Unfolded basis image
                m = m + 1;
            end
        end
    end
    im{z} = Ibasis;
    disp(z);
end


mean_sunset = uint8(mean(featvecmat))
m = 1;
for i = 1:3
    for j = 1:64
        for k = 1:64
            im_mean_sunset(k,j,i) = uint8(mean_sunset(m));  % Unfolded basis image
            m = m + 1;
        end
    end
end


% Show faces

imagearr = cat(4,im_mean_sunset,im{1:99});
montage(imagearr, 'DisplayRange',[0,255]);


%% Determining the maximum tolerable reconstruction error with a finite number of pca components

for O = 1:size(featvecmat,1)
    
    % For the Oth image
    mean_sunset = double(mean(featvecmat));
    imvecO = featvecmat(O,:)'- double(mean_sunset');
    proj_vecO = coeff(:,1:num_components)'*imvecO;            % Project the i first principle components
    newfeatvec(i,:) = proj_vecO;                              % Project features into new feature space
    
    imavec_reconstructed = double(mean_sunset');
    for mmm = 1:num_components
        imavec_reconstructed = imavec_reconstructed + proj_vecO(mmm)*coeff(:,mmm);
    end
    
    
    error_reconstruct(O) = sum((featvecmat(O,:)'-imavec_reconstructed).^2);
    
    
    
    if(mod(O,50)==0)
       disp(O) 
    end
end

errmax = max(error_reconstruct)





%%
O = 1;
% For the Oth image 
mean_sunset = uint8(mean(featvecmat))
imvecO = featvecmat(O,:)'- double(mean_sunset');
proj_vecO = coeff(:,num_components)'*imvecO;            % Project the i first principle components


imavec_reconstructed = double(mean_sunset');
for mmm = 1:num_components
    imavec_reconstructed = imavec_reconstructed + proj_vecO(mmm)*coeff(:,mmm);
end


error_reconstruct = sum((featvecmat(O,:)'-imavec_reconstructed).^2)


if error_reconstruct< error_threshold
   face = 1; % Image is face if it was reconstructed with low error from the n principle components
else
   face = 0;
end


% LDA classifier



% KNN Classifier



% SVM
