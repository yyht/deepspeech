odpscmd=$1
model_folder=$2
model_zip=$3

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*"

pai_command="
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
pai -name tensorflow1120
    -Dscript='file://${model_zip}'
    -DentryFile='./deepspeech/run_wave2vec_conformer.py' 
    -DjobName='bert_qqp'
    -Dtags='bert'
    -DmaxHungTimeBeforeGCInSeconds=0
    -DmaxTrainingTimeInHour=1440
    -DautoStrategy='false'
    -Dcluster='{\"worker\":{\"count\":10, \"gpu\":100}}'
    -DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/deepspeech/conformer_params_reduced_length_pretraining'
    -Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."

