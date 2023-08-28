REPO_DIR="domino"
WORKSPACE_DIR=/apdcephfs_cq3/share_1603164/user/lfsong/exp.tencent_chat/data/ppo_train_test

MEGATRON_REPO=${WORKSPACE_DIR}/${REPO_DIR}
export PYTHONPATH=${MEGATRON_REPO}:$PYTHONPATH

python3 ${MEGATRON_REPO}/examples/train_domino_dummy.py