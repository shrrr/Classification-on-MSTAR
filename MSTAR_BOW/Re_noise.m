OriginalData_path=...
    {'C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\DataSet\train','C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\DataSet\test'};
ProcessData_path=...
    {'C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\MSTAR_BOW\preprocess_train_2','C:\Users\Eric\Desktop\Target-Detection-in-MSTAR-Images-master\MSTAR_BOW\preprocess_test_2'};
foldList={'2S1','BRDM_2','BTR_60','D7','SN_132','SN_9563','SN_C71','T62','ZIL131','ZSU_23_4'};

for lj=1:length(OriginalData_path)
for i=1:length(foldList)
    foldTemp=fullfile(OriginalData_path{lj},foldList{i});
    foldTemp2=fullfile(ProcessData_path{lj},foldList{i});
    mkdir(foldTemp2);
    TempList=dir(foldTemp);
    for j=1:length(TempList)
        if length(TempList(j).name)>4 & ((TempList(j).name(end-3:end)=='.jpg')|(TempList(j).name(end-3:end)=='.JPG'))
            I1_path=fullfile(foldTemp,TempList(j).name);
            I2_path=fullfile(foldTemp2,TempList(j).name);
            I1=imread(I1_path);
            [m,n]=size(I1);
            Y=dct2(I1); 
            I=zeros(m,n);
            %高频屏蔽
            I(1:round(m/7),1:round(n/7))=1; 
            Ydct=Y.*I;
            %逆DCT变换
            I2=uint8(idct2(Ydct));
            f_average = fspecial('average',[5 5]);
            I2 = imfilter(I2,f_average);
            imwrite(I2,I2_path);
        end     
    end   
end
end
