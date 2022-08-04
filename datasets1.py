import os
import gzip
import logging
import requests


import logging.config
logging.config.fileConfig(fname='./config/log.init', disable_existing_loggers=False)
log = logging.getLogger('system')

class Loader:
    URL = 'http://kdd.ics.uci.edu/databases/kddcup99/'
    SAVE_PATH = './datasets/'

    def __init__(self) -> None:
        if not os.path.exists(Loader.SAVE_PATH + 'kddcup.data_10_percent'):
            self.download('kddcup.data_10_percent')
        if not os.path.exists(Loader.SAVE_PATH + 'corrected'):
            self.download('corrected')

    def download(self, filename):
        # 下载数据
        log.info(f'>>> Start downloading {filename}.gz from {Loader.URL}')
        url = Loader.URL + filename + '.gz'
        rsp = requests.get(url)
        if rsp.status_code != 200:
            raise Exception(f'>>> Failed to download data from {url}, http status: {rsp.status_code}')
        with open(Loader.SAVE_PATH + filename + '.gz', 'wb') as f:
            f.write(rsp.content)
            f.close()
        # 解压数据
        log.info(f'>>> Start unzipping the file {filename}.gz')
        zip = gzip.GzipFile(Loader.SAVE_PATH + filename + '.gz')    
        open(Loader.SAVE_PATH + filename, "wb+").write(zip.read())
        zip.close()
        # 删除无用压缩包
        os.remove(Loader.SAVE_PATH + filename + '.gz')
        log.info(f'Success to downloading {Loader.SAVE_PATH}{filename}')


if __name__ == '__main__':
    loader = Loader()