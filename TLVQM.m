%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute features for a set of video files from LIVE-Qualcomm databse

% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto


% Read subjective data
mos_data=load('D:\JAVERIANA\THESIS\ALGORITMO_TLVQM\nr-vqa-consumervideo-master\matrices qualcomm\mos_qualcomm_expositure.mat');
mos_data_biased=mos_data.biased;
mos_data_unbiased=mos_data.unbiased;

% Direccion de la carpeta de videos
videos_path='E:\ALEJANDRO LEDESMA\THESIS\LIVE-Qualcomm\Exposure\AVI\';

% Direccion de la carpeta donde se guardaran los resultados extraidos 
features_path='D:\JAVERIANA\THESIS\RESULTS\TLVQM\Features\Exposure\';

% Open feature file for output
feature_file = '.\LIVE_features.csv'; 
fid_ftr = fopen(feature_file,'w+');
list=dir(videos_path)
% Iterador de la fila en la que guardara las caracteristicas en el archivo de excel 
i=5;
% Loop through all the video files in the database
for z=7:length(list)
% Se eliminan los archivos yuv para evitar leer archivos incorrectos 
        delete *.yuv

z
    full_yuv_path = sprintf('%s%s', videos_path,list(z).name);
    
    % Para conocer la cantidad de frames de cada video y su resolucion                   
    Video=full_yuv_path;
    v = VideoReader(Video);
    vidHeight = v.Height;
    vidWidth = v.Width;
    reso = [vidWidth vidHeight];

    yuv_name=   'out9.yuv'; 

    % Convierte los archivos de video de formato Avi a Yuv 
    mm2yuv(full_yuv_path,yuv_name);%convierte el video de avi a yuv 

    % Compute features for each video file
    fprintf('Computing features for sequence: %s\n',full_yuv_path);
    tic
    features = compute_nrvqa_features(yuv_name, reso, 30);
    toc
    file_path=strcat(list(z).name,'.mat');
    file_name=strcat(features_path,file_path);
    save (file_name,'features','-v7.3');

    % Write features to csv file for further processing
    fprintf(fid_ftr, '%2.2f, %2.2f,%0.2f,%0.2f', ...
            mos_data_unbiased(i), ...
            mos_data_biased(i), reso(1)/1920, 1);
    for j=1:length(features)
        fprintf(fid_ftr, ',%0.5f', features(j));
    end
    fprintf(fid_ftr, '\n');
    
    delete *.yuv
      i=i+1;
end
fclose(fid_ftr);
fprintf('All done!\n');
