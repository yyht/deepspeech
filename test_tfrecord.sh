odpscmd=$1
model_folder=$2
model_zip=$3

if [ ! -f ${model_zip} ]
then
  rm ${model_zip}
fi

zip -r ${model_zip} ${model_folder} -x "*.DS_Store,*.git*"

pai_command="
# set odps.running.cluster=AY100G;
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
pai -name tensorflow1120
    -project algo_public
    -Dscript='file://${model_zip}'
    -Dcluster='{\"worker\":{\"count\":100, \"gpu\":0, \"cpu\":100, \"memory\":2000}}'
    -DentryFile='./deepspeech/test_tfrecord.py' 
    -DhyperParameters='file:///Users/xuhaotian/Desktop/my_work/deepspeech/test_tfrecord_params'
    -Dbuckets='oss://alg-misc/BERT/?role_arn=acs:ram::1265628042679515:role/yuefeng2&host=cn-hangzhou.oss-internal.aliyun-inc.com'
;
"
echo "${pai_command}"
${odpscmd} -e "${pai_command}"
echo "finish..."

