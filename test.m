function test()
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.

I = imread('images/original/or_6.jpg');

I = rgb2gray(I);

imshow(I)
title('Original Image')

mask = zeros(size(I));
mask(25:end-25,25:end-25) = 1;
figure
imshow(mask)
title('Initial Contour Location')

bw = activecontour(I,mask,600);

figure
imshow(bw)
title('Segmented Image')

end