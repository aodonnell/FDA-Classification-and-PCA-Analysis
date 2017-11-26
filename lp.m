%%


% 
im = imread('fabric.png');
im = imresize(im,[64 64]);
featvecmat(2,:) = im(:);
num_components = 570
tic
for O = 1:500
    
    % For the Oth image
    mean_sunset = (mean(featvecmat));
    imvecO = featvecmat(O,:)'- double(mean_sunset');
    proj_vecO = coeff(:,1:num_components)'*imvecO;            % Project the i first principle components
    
    
    imavec_reconstructed = double(mean_sunset');
    for mmm = 1:num_components
        imavec_reconstructed = imavec_reconstructed + proj_vecO(mmm)*coeff(:,mmm);
    end
    
    
    error_reconstruct(O) = sum((featvecmat(O,:)'-imavec_reconstructed).^2);
end
    

%%
error_threshold = 2.06*10^6;        %99.5% of images of faces were classified correctly (This was to take outlyers into account)
if error_reconstruct< error_threshold
    face = 1; % Image is face if it was reconstructed with low error from the n principle components
else
    face = 0;
end


% LDA classifier



% KNN Classifier



% SVM
