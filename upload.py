import fire
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import glob
import os


class Upload:
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)

    def file(self, name):
        file_ = self.drive.CreateFile({f'title': name.split('/')[-1]})
        file_.SetContentFile(name)
        file_.Upload()
        print('title: %s, id: %s' % (file_['title'], file_['id']))

    def files(self, path, regex=None):
        if regex is None:
            regex = '*'
        for f in glob.glob(os.path.join(path, regex)):
            self.file(f)


if __name__ == '__main__':
    fire.Fire(Upload)
