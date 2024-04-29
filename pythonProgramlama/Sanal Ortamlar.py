# Sanal Ortam ve Paket Yönetimi #

# Sanal Ortamların Listelenmesi
# conda env list

# Sanal ortam oluşturma
# conda create -n myenv

# Sanal ortamı aktif etme
# conda activate myenv

# Yüklü paketlerin listelenmesi
# conda list

# Paket Yükeleme
# conda install numpy

# Birden fazla paket yükleme
# conda install numpy scipy pandas

# Paket Silme
# conda remove package_name

# Belirli birr versiyona göre paket yükleme
# conda install numpy=1.20.1

# Paket yükseltme
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi
# conda upgrade -all

# pip: pypi (python package index) paket yönetim aracı
# pip install pandas
# pip install pandas==1.2.1

# Dışarı aktarma ve paylaşma
# conda env export > environment.yaml