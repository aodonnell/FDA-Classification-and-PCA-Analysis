clear
clc


% PCA
a = dir('celeba_low/*.jpg')

for i = 1 : size(a,1)       
   I = imread(['celeba_low/',a(i).name]);
   featvecmat(i,:) = I(:);       % Transform image into feature vector. Do this for each image
end


featvecmat = double(featvecmat);            %Convert features to doubles so it can be used in MATLAB's pca function
[coeff, score, latent] = pca(featvecmat);

var_captured = 0;           % Variance captured from features (between 0 and 1)
i = 1;
while (i<size(coeff,2) && var_captured<0.98) %Loop through until 98% of variance is captured by eigen vectors
    var_captured = sum(latent(1:i))/sum(latent);               % Proportion of variance up to the ith eigen vector of pca 
    i = i + 1;
end
num_components = i;

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
