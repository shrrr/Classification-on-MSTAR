close all
I2 = imread('E:\CurrentFiles\Documents\研一上课程\雷达目标识别\Target-Detection-in-MSTAR-Images-master\DataSet\train\2S1\HB19377.JPG');
%% SURF Method
% [height,width,numChannels] = size(I);
% gridStep = 8; % in pixels
% gridX = 1:gridStep:width;
% gridY = 1:gridStep:height;
% [x,y] = meshgrid(gridX, gridY);
% gridLocations = [x(:) y(:)];
% multiscaleGridPoints = [SURFPoints(gridLocations, 'Scale', 1.6); 
%                         SURFPoints(gridLocations, 'Scale', 3.2);
%                         SURFPoints(gridLocations, 'Scale', 4.8);
%                         SURFPoints(gridLocations, 'Scale', 6.4)];
% % corners   = detectSURFFeatures((I2));
% % strongest = selectStrongest(corners, 3);
% [hog2, validPoints, ptVis] = extractHOGFeatures(I2, multiscaleGridPoints);
% % [surf,validpoints]=extractFeatures(I2,strongest,'Method','SURF');
% figure;
% imshow(I2); hold on;
% plot(ptVis,'color','g');
% var(features,[],2)
%% HOG Method
% [featureVector,hogVisualization] = extractHOGFeatures(I2);
% figure;
% imshow(I2); 
% hold on;
% plot(hogVisualization);

%% reduce noise
% -------------- DCT ----------------
subplot(121); 
imshow(I2);
[m,n]=size(I2);
Y=dct2(I2); 
I=zeros(m,n);
%高频屏蔽
I(1:round(m/3),1:round(n/3))=1; 
Ydct=Y.*I;
%逆DCT变换
Y=uint8(idct2(Ydct)); 
f_average = fspecial('average',[5 5]);
f_prewitt=fspecial('prewitt');
f_sobel = fspecial('sobel')
% f_laplacian = fspecial('laplacian',0)
% Y = edge(Y,'sobel'); 
Y = imfilter(Y,f_average);
% Y = imfilter(Y,f_sobel);
% Y = imfilter(Y,f_laplacian);
% Y = imfilter(Y,f_prewitt);
%结果输出
subplot(122);
imshow(Y);

% --------------- little wave -------------
% %用小波函数coif2对图像XX进行2层
% % 分解
% XX = I2;
% [c,l]=wavedec2(XX,2,'coif2'); 
% % 设置尺度向量
% n=[1,2];                  
% % 设置阈值向量 , 对高频小波系数进行阈值处理
% p=[10.28,24.08]; 
% nc=wthcoef2('h',c,l,n,p,'s');
% % 图像的二维小波重构
% X1=waverec2(nc,l,'coif2');   
% subplot(223);              
% imshow(uint8(X1));                
% %colormap(map);            
% title(' 第一次消噪后的图像 '); 
% %再次对高频小波系数进行阈值处理
% mc=wthcoef2('v',nc,l,n,p,'s');
% % 图像的二维小波重构
% X2=waverec2(mc,l,'coif2');  
% subplot(224);             
% imshow(uint8(X2));               
% title(' 第二次消噪后的图像 ');   

I=Y;
[height,width,numChannels] = size(I);
gridStep = 8; % in pixels
gridX = 1:gridStep:width;
gridY = 1:gridStep:height;
[x,y] = meshgrid(gridX, gridY);
gridLocations = [x(:) y(:)];
% multiscaleGridPoints = [SURFPoints(gridLocations, 'Scale', 1.6); 
%                         SURFPoints(gridLocations, 'Scale', 3.2);
%                         SURFPoints(gridLocations, 'Scale', 4.8);
%                         SURFPoints(gridLocations, 'Scale', 6.4)];
multiscaleKAZEPoints = detectKAZEFeatures(I);

% corners   = detectSURFFeatures((I2));
% strongest = selectStrongest(corners, 3);
[features, valid_points]= extractFeatures(I, multiscaleKAZEPoints,'Upright',true);
figure; imshow(I); hold on;
plot(valid_points.selectStrongest(10),'showOrientation',true);
