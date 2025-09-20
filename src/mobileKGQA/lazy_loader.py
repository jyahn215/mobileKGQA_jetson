import h5py
import pickle
import numpy as np


class LazyDataset:
    """
    HDF5에 pickle로 저장된 배치를 lazy-loading 하는 단순 로더.
    - get_batch(idx): 현재 순서(order) 기준 idx번째 배치 반환
    - reset_batches(is_sequential): 인덱스 순서 재설정(순차/무작위)
    """

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self._file = h5py.File(h5_path, "r", swmr=True)
        self.num_data = int(self._file["num_data"][()])
        self.max_local_entity = int(self._file["max_local_entity"][()])

        # batch_0, batch_1, ... 형태의 key 정렬
        self.keys = sorted(
            [k for k in self._file.keys() if k not in ("num_data", "max_local_entity")],
            key=lambda x: int(x.split("_")[1]),
        )

        # 현재 인덱스 순서 (기본은 0~N-1 순차)
        self._order = np.arange(len(self.keys))

    def get_batch(self, idx: int, batch_size=None, fact_dropout=0., test=False): # dummy parameters for compatibility
        """
        현재 설정된 순서(_order)에서 idx번째 배치를 반환
        """
        real_idx = self._order[idx]
        key = self.keys[real_idx]
        raw_bytes = bytes(self._file[key][()])
        return pickle.loads(raw_bytes)

    def reset_batches(self, is_sequential: bool = True):
        """
        배치 순서 재설정
        - is_sequential=True  : 0,1,2,... 순차
        - is_sequential=False : 무작위 셔플
        """
        if is_sequential:
            self._order = np.arange(len(self.keys))
        else:
            self._order = np.random.permutation(len(self.keys))

    def __len__(self):
        return len(self.keys)

    def close(self):
        """HDF5 파일 닫기"""
        self._file.close()


if __name__ == "__main__":
    dataset = LazyDataset(
        "./../../data/preprocessed_data/webqsp/_domains/total/test_batches_relbert_256_8.h5"
    )

    print("num_data:", dataset.num_data, len(dataset))
