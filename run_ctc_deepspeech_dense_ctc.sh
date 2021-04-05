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
    -DentryFile='./deepspeech/run_ctc_deepspeech.py' 
    -DjobName='bert_qqp'
    -Dtags='bert'
    -DmaxHungTimeBeforeGCInSeconds=0
    -DmaxTrainingTimeInHour=1440
    -DautoStrategy='false'
	-Dcluster='{\"worker\":{\"count\":5, \"gpu\":100}}'
    -DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/deepspeech/deepspeech_params_dense_ctc'
    -Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com';
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."

