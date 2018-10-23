% Generating images for Data set

% Reading all necessary image files and resizing them 
img1 = imresize(imread('dollar_train.jpg'), [820,680]);
img2 = imresize(imread('Pound_train.PNG'), [820,680]);
img3 = imresize(imread('Euro_train.PNG'), [820,680]);
img4 = imresize(imread('Rupee_train.PNG'), [820,680]);
img5 = imresize(imread('Yen_train.PNG'), [820,680]);

img = [img1;img2;img3;img4;img5]; % Creating an array of images

gray = rgb2gray(img); % Converting to gray scale image

bin = imbinarize(gray); % Converting to binary image

BW = edge(gray, 'Canny', 0.4); % Using Canny edge detection 

fill = imfill(BW, 'holes'); % Filling holes after edge detection

% Used to dilate the image or increase pixel size of characters detected
se1 = strel('line',2,0);
se2 = strel('line',2,90);
composition = imdilate(fill,[se1 se2],'full');

% Bounding box for all characters 
box = regionprops(composition,'Area', 'BoundingBox');
len = length(box);

bb = struct2dataset(box);

% Saving Area of Bounding Boxes as matrix
Area = bb(1:length(bb),1);
Area = dataset2cell(Area);
Area = Area(2:length(Area),1);
Area = cell2mat(Area);

% Saving Coordinates of Bounding Boxes as matrix
Coordinates = bb(1:length(bb),2);
Coordinates = dataset2cell(Coordinates);
Coordinates = Coordinates(2:length(Coordinates),1);
Coordinates = cell2mat(Coordinates);

% Appending Area and Coordinates as one matrix
data = [Area Coordinates];
data = sortrows(data, 3, 'ascend');

% Bounding boxes will be displayed on a gray scale image and
% images will be extracted
imshow(gray);
hold on
count = 0;

% Images will be rotated by 0,5,-5,10,-10 degrees to increase data set
rotation = [0, 5, -5, 10, -10];
rot_len = length(rotation);

% Following loop is used to map Bounding boxes to gray scale image
% All these boxes are extracted as images
% Images are rotated by mentioned rotation angles and saved 
% Images are saved as img#####.jpg
for c = 1:len
    if data(c,1) >= 100
        bx = data(c,2:5);
        rectangle('Position',bx);
        crop = imcrop(gray,bx);
        crop = imresize(crop,[28,28]);
        
        for rot_degree = 1:rot_len
            rot = imrotate(crop, rotation(rot_degree));
            [r,c]= size(rot);
            for i = 1:r
                for j = 1:c
                    if (rot(i,j) == 0)  
                        rot(i,j) = 255;
                    else
                        rot(i,j) = rot(i,j);
                    end
                end
            end
        
            
        rot = imresize(rot, [28,28]);
        count = count + 1;
        mm=num2str(count);
        
        % Used to format image name
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
        
        % Saving images
        imwrite(rot,strcat('img',num2str(mm1),'.jpg'));
        
        end
    end    
end

% Total count displayed is 10000
disp(count);
