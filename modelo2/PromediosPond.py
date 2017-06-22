#####################################################################################
# 1.- PROMEDIOS PONDERADOS POR 1-max(tst_loss,val_loss,loss) de los 25 mejores modelos:
#####################################################################################
import pandas as pd
pd.options.display.max_colwidth = 180
df2=pd_read_csv(s_output_path + 'rets_full.csv')
df2 = df2[df2['tst_loss'] + df2['val_loss'] + df2['loss'] < 3 * 0.82]
df2['sum_val'] = df2['tst_loss'] + df2['val_loss'] + df2['loss']
df2['max_val'] = df2[['tst_loss', 'val_loss', 'loss']].max(axis=1)
df2['min_val'] = df2[['tst_loss', 'val_loss', 'loss']].min(axis=1)
pesos = df2[df2['max_val'] < 0.8001].sort_values('max_val', ascending=False)['fich'] # 25 ficheros (weights)
vals =  1 - df2[df2['max_val'] < 0.8001].sort_values('max_val', ascending=False)['max_val'] # 25 ficheros (weights)
mods  = pd_DataFrame([a[0].replace('\\', '/').replace(s_output_path, '') for a in [glob_glob(s_output_path + 'modeloprev_*' + fichhdf5[-20:-5] + '.json') for fichhdf5 in list(pesos)]], columns = ['modelo'])
assert(all(mods.count()==pesos.count()))
assert(all(mods.count()==vals.count()))
mods['pesos']=list(pesos)
mods['pond_val']=list(vals)
test_data = np_load(s_input_path + 'test' + s_filename_suffix + '.npy')
test_id = np_load(s_input_path + 'test_id' + s_filename_suffix + '.npy')
for idx, reg in mods.sort_values('modelo').iterrows():
  fichjson = reg['modelo']
  assert(os_path_isfile(s_output_path + fichjson))
  fichhdf5 = reg['pesos']
  assert(os_path_isfile(s_output_path + fichhdf5))
  #print(fichjson, fichhdf5, 'Ok.')

fichjson, model, df, tot_pond = ['', None, None, 0]
for idx, reg in mods.sort_values('modelo').iterrows():
  fichhdf5 = reg['pesos']
  tmp_pond = reg['pond_val']
  if fichjson != reg['modelo']:
    fichjson = reg['modelo']
    mi_batchsize = int(fichjson[4+fichjson.find('bch-'):fichjson.find('_', 1+fichjson.find('bch-'))]) # batchsize a partir del nombre del json!!!
    #assert(os_path_isfile(s_output_path + fichjson))
    model = leer_pesos_modelo(leer_modelo_json(fichjson), mi_loss, mi_optimizador, mis_metrics, fichero_pesos = fichhdf5)
  else:
    model = leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, fichero_pesos = fichhdf5)
  
  assert(n_resize_to == model.get_config()[0]['config']['batch_input_shape'][-1]) # Asegurarse de que x_train.shape es la correcta! (32, 64 o 256)
  # Creamos preds:
  pred = model.predict_proba(test_data)
  dftmp = pd_DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
  if not df is None:
    df       += tmp_pond * dftmp
    tot_pond += tmp_pond
  else:
    df       = tmp_pond * dftmp
    tot_pond = tmp_pond
  
  df['TmpType_1'] = dftmp[['Type_1','Type_2','Type_3']].max(axis=1) == dftmp['Type_1']
  df['TmpType_2'] = dftmp[['Type_1','Type_2','Type_3']].max(axis=1) == dftmp['Type_2']
  df['TmpType_3'] = dftmp[['Type_1','Type_2','Type_3']].max(axis=1) == dftmp['Type_3']
  df['AllType_1'] = df[['Type_1','Type_2','Type_3']].max(axis=1) == df['Type_1']
  df['AllType_2'] = df[['Type_1','Type_2','Type_3']].max(axis=1) == df['Type_2']
  df['AllType_3'] = df[['Type_1','Type_2','Type_3']].max(axis=1) == df['Type_3']
  print(df[['TmpType_1', 'AllType_1', 'TmpType_2', 'AllType_2', 'TmpType_3', 'AllType_3']].mean())

# Finalmente, promediamos y creamos submit:
df /= tot_pond
s_unique_filename = jj_datetime_filename()
df.insert(0, 'image_name', test_id) # df['image_name'] = test_id
# df = df.sort_values('image_name')
s_filename_submit = 'submit' + s_filename_suffix + '_{:.4f}-{:.4f}-avg{:}_'.format((1-vals).min(), (1-vals).max(), len(vals)) + s_unique_filename + '.csv' # submit_32_0.85949_20170514_111127.csv
print(jj_datetime(), 'Creamos submit.csv... [' + s_filename_submit + ']')
# Guardamos ficheros (submit, hdf5 y modelo):
df.to_csv(s_output_path + s_filename_submit, index=False) # submit_256_0.7349-0.8001-avg25_20170531_175312.csv

#####################################################################################
# 3.- Corregimos errores:
#####################################################################################
s_filename_submit='submit_256_0.7349-0.8001-avg25_20170531_175312.csv'
df3=pd_read_csv(s_output_path + s_filename_submit)
df3=df3[['image_name','Type_1','Type_2','Type_3']]
[len(df3[df3.ix[:,1] > 1]), len(df3[df3.ix[:,2] > 1]), len(df3[df3.ix[:,3] > 1])]
[len(df3[df3.ix[:,1] < 0]), len(df3[df3.ix[:,2] < 0]), len(df3[df3.ix[:,3] < 0])]
#df3.ix[[1,46,293,300,337,407],2]=1 # Eran 1.0000001xxxx y el concurso no acepta valores > 1
df3.to_csv(s_output_path + s_filename_submit, index=False)

s_filename_submit='submit_256_0.7349-0.8001-avg22_20170531_182728.csv'
df3=pd_read_csv(s_output_path + s_filename_submit)
df3=df3[['image_name','Type_1','Type_2','Type_3']]
[len(df3[df3.ix[:,1] > 1]), len(df3[df3.ix[:,2] > 1]), len(df3[df3.ix[:,3] > 1])]
[len(df3[df3.ix[:,1] < 0]), len(df3[df3.ix[:,2] < 0]), len(df3[df3.ix[:,3] < 0])]
df3[df3.ix[:,2] > 1].index
#df3.ix[[1, 46, 293, 300, 326, 337, 388, 407, 417],2]=1 # Eran 1.0000001xxxx y el concurso no acepta valores > 1
df3.to_csv(s_output_path + s_filename_submit, index=False)
