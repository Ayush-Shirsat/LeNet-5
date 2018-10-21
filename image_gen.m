clear all
close all
clc

img1 = imresize(imread('dollar_train.jpg'), [820,680]);
img2 = imresize(imread('Pound_train.PNG'), [820,680]);
img3 = imresize(imread('Euro_train.PNG'), [820,680]);
img4 = imresize(imread('Rupee_train.PNG'), [820,680]);
img5 = imresize(imread('Yen_train.PNG'), [820,680]);
img = [img1;img2;img3;img4;img5];
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
            mm1=strcat('0000',mm);
        elseif count < 100
            mm1=strcat('000',mm);
        elseif count < 1000
            mm1=strcat('00',mm);
        elseif count < 10000
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(crop,strcat('img',num2str(mm1),'.jpg'));
        
        rot5 = imrotate(crop, 5);
        [r,c]= size(rot5);
        for i = 1:r
            for j = 1:c
               if (rot5(i,j) == 0)  
                   rot5(i,j) = 255;
               else
                   rot5(i,j) = rot5(i,j);
               end
            end
        end
        count = count + 1;
        rot5 = imresize(rot5, [28,28]);
        mm=num2str(count);
        if count < 10
            mm1=strcat('0000',mm);
        elseif count < 100
            mm1=strcat('000',mm);
        elseif count < 1000
            mm1=strcat('00',mm);
        elseif count < 10000
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(rot5,strcat('img',num2str(mm1),'.jpg'));
        
        rotm5 = imrotate(crop, -5);
        [r,c]= size(rotm5);
        for i = 1:r
            for j = 1:c
               if (rotm5(i,j) == 0)  
                   rotm5(i,j) = 255;
               else
                   rotm5(i,j) = rotm5(i,j);
               end
            end
        end
        count = count + 1;
        rotm5 = imresize(rotm5, [28,28]);
        mm=num2str(count);
        if count < 10
            mm1=strcat('0000',mm);
        elseif count < 100
            mm1=strcat('000',mm);
        elseif count < 1000
            mm1=strcat('00',mm);
        elseif count < 10000
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(rotm5,strcat('img',num2str(mm1),'.jpg'));
        
        rot10 = imrotate(crop, 10);
        [r,c]= size(rot10);
        for i = 1:r
            for j = 1:c
               if (rot10(i,j) == 0)  
                   rot10(i,j) = 255;
               else
                   rot10(i,j) = rot10(i,j);
               end
            end
        end
        count = count + 1;
        rot10 = imresize(rot10, [28,28]);
        mm=num2str(count);
        if count < 10
            mm1=strcat('0000',mm);
        elseif count < 100
            mm1=strcat('000',mm);
        elseif count < 1000
            mm1=strcat('00',mm);
        elseif count < 10000
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(rot10,strcat('img',num2str(mm1),'.jpg'));
        
        rotm10 = imrotate(crop, 10);
        [r,c]= size(rotm10);
        for i = 1:r
            for j = 1:c
               if (rotm10(i,j) == 0)  
                   rotm10(i,j) = 255;
               else
                   rotm10(i,j) = rotm10(i,j);
               end
            end
        end
        count = count + 1;
        rotm10 = imresize(rotm10, [28,28]);
        mm=num2str(count);
        if count < 10
            mm1=strcat('0000',mm);
        elseif count < 100
            mm1=strcat('000',mm);
        elseif count < 1000
            mm1=strcat('00',mm);
        elseif count < 10000
            mm1=strcat('0',mm);
        else 
            mm1=strcat(mm);
        end
        imwrite(rotm10,strcat('img',num2str(mm1),'.jpg'));
    end
end
disp(count);
