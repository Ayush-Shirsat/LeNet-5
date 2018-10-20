clear all
close all
clc

img = imread('dollar_train.jpg');
% imshow(img);

gray = rgb2gray(img);
% imshow(gray);

bin = imbinarize(gray);
% imshow(bin);

BW = edge(gray, 'Canny', 0.10);
% imshow(BW);

fill = imfill(BW, 'holes');
% figure,
% imshow(fill);

se1 = strel('line',2,0);
se2 = strel('line',2,90);
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
    if data(c,1) >= 100
        count = count + 1;
        bx = data(c,2:5);
        rectangle('Position',bx);
        crop = imcrop(gray,bx);
        crop = imresize(crop,[28,28]);
        mm=num2str(count);
        if count < 10
            mm1=strcat('00',mm);
        elseif count < 100
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(crop,strcat('img',num2str(mm1),'.jpg'));
    end
end
disp(count);
