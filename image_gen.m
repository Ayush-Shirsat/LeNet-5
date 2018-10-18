clear all
close all
clc

img = imread('dataset.PNG');
% imshow(img);

gray = rgb2gray(img);
% imshow(gray);

bin = imbinarize(gray);
% imshow(bin);

BW = edge(gray, 'Canny', 0.4);
%imshow(BW);

fill = imfill(BW, 'holes');
% figure,
% imshow(fill);

se1 = strel('line',3,0);
se2 = strel('line',3,90);
composition = imdilate(fill,[se1 se2],'full');
% imshow(composition);

box = regionprops(composition,'Area', 'BoundingBox');
len = length(box);

bb = struct2dataset(box);

Area = bb(1:length(bb),1);
Area = dataset2cell(Area);
Area = Area(2:length(Area),1);
Area = cell2mat(Area);

Coordinates = bb(1:length(bb),2);
Coordinates = dataset2cell(Coordinates);
Coordinates = Coordinates(2:length(Coordinates),1);
Coordinates = cell2mat(Coordinates);

data = [Area Coordinates];
data = sortrows(data, 3, 'ascend');
% figure,
imshow(gray);
hold on
count = 0;
for c = 1:len
    if data(c,1) >= 200
        count = count + 1;
        bx = data(c,2:5);
%         disp(bx);
        rectangle('Position',bx);
        crop = imcrop(gray,bx);
        crop = imresize(crop,[28,28]);
        imwrite(crop,strcat('img',num2str(count),'.jpg'));
    end
end
disp(count);

%show montage
fileFolder = 'C:\Users\Ayush Shirsat\Desktop\Mini_proj';
dirOutput = dir(fullfile(fileFolder,'*.jpg')); % *.jpg indicates 
fileNames = {dirOutput.name};
figure,
montage(fileNames,'Size',[10 15]);
