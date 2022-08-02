import os
import csv
import logging
import numpy as np

from enum import Enum, unique

log = logging.getLogger('system')

@unique
class Procotol(Enum):
    # unknow  = 0
    tcp     = 1
    udp     = 2
    icmp    = 3

    def fromName(name: str):
        for protocol in Procotol:
            if protocol.name == name:
                return protocol
        return Procotol.unknow
    
    def fromValue(value: int):
        for protocol in Procotol:
            if protocol.value == value:
                return protocol
        return Procotol.unknow

@unique
class Service(Enum):
    '''目标主机的网络服务类型'''
    # unknow = 0
    aol =  1
    auth =  2
    bgp =  3
    courier =  4
    csnet_ns =  5
    ctf =  6
    daytime =  7
    discard =  8
    domain =  9
    domain_u =  10
    echo =  11
    eco_i =  12
    ecr_i =  13
    efs =  14
    exec =  15
    finger =  16
    ftp =  17
    ftp_data =  18
    gopher =  19
    harvest =  20
    hostnames =  21
    http =  22
    http_2784 =  23
    http_443 =  24
    http_8001 =  25
    imap4 =  26
    IRC =  27
    iso_tsap =  28
    klogin =  29
    kshell =  30
    ldap =  31
    link =  32
    login =  33
    mtp =  34
    name =  35
    netbios_dgm =  36
    netbios_ns =  37
    netbios_ssn =  38
    netstat =  39
    nnsp =  40
    nntp =  41
    ntp_u =  42
    other =  43
    pm_dump =  44
    pop_2 =  45
    pop_3 =  46
    printer =  47
    private =  48
    red_i =  49
    remote_job =  50
    rje =  51
    shell =  52
    smtp =  53
    sql_net =  54
    ssh =  55
    sunrpc =  56
    supdup =  57
    systat =  58
    telnet =  59
    tftp_u =  60
    tim_i =  61
    time =  62
    urh_i =  63
    urp_i =  64
    uucp =  65
    uucp_path =  66
    vmnet =  67
    whois =  68
    X11 =  69
    Z39_50 =  70

    def fromName(name: str):
        for service in Service:
            if service.name == name:
                return service
        return None
    
    def fromValue(value: int):
        for service in Service:
            if service.value == value:
                return service
        return None

@unique
class Flag(Enum):
    '''连接正常或错误的状态'''
    OTH     = 0
    REJ     = 1
    RSTO    = 2
    RSTOS0  = 3
    RSTR    = 4
    S0      = 5
    S1      = 6
    S2      = 7
    S3      = 8
    SF      = 9
    SH      = 10

    def fromName(name: str):
        for flag in Flag:
            if flag.name == name:
                return flag
    
    def fromValue(value: int):
        for flag in Flag:
            if flag.value == value:
                return flag

@unique
class Assail(Enum):
    NORMAL  = 0
    DOS     = 1
    R2L     = 2
    U2R     = 3
    PROBE   = 4

    @staticmethod
    def toAssail(attack):
        relation = {
            'normal'            : Assail.NORMAL,
            'ipsweep'            : Assail.PROBE,
 	        'mscan'             : Assail.PROBE,
 	        'nmap'              : Assail.PROBE,
 	        'portsweep'         : Assail.PROBE,
 	        'saint'             : Assail.PROBE,
 	        'satan'             : Assail.PROBE,
            'apache2'           : Assail.DOS,
            'back'              : Assail.DOS,
            'land'              : Assail.DOS,
            'mailbomb'          : Assail.DOS,
            'neptune'           : Assail.DOS,
            'pod'               : Assail.DOS,
            'processtable'      : Assail.DOS,
            'smurf'             : Assail.DOS,
            'teardrop'          : Assail.DOS,
            'udpstorm'          : Assail.DOS,
            'buffer_overflow'   : Assail.U2R,
            'httptunnel'        : Assail.U2R,
            'loadmodule'        : Assail.U2R,
            'perl'              : Assail.U2R,
            'ps'                : Assail.U2R,
            'rootkit'           : Assail.U2R,
            'sqlattack'         : Assail.U2R,
            'xterm'             : Assail.U2R,
            'ftp_write'         : Assail.R2L,
            'guess_passwd'      : Assail.R2L,
            'imap'              : Assail.R2L,
            'multihop'          : Assail.R2L,
            'named'             : Assail.R2L,
            'phf'               : Assail.R2L,
            'sendmail'          : Assail.R2L,
            'snmpgetattack'     : Assail.R2L,
            'snmpguess'         : Assail.R2L,
            'spy'               : Assail.R2L,
            'warezclient'       : Assail.R2L,
            'warezmaster'       : Assail.R2L,
            'worm'              : Assail.R2L,
            'xlock'             : Assail.R2L,
            'xsnoop'            : Assail.R2L
        }
        return relation[attack]

@unique
class Attack(Enum):
    normal			= 0
    ipsweep			= 1
    mscan			= 2
    nmap			= 3
    portsweep		= 4
    saint			= 5
    satan			= 6
    apache2			= 7
    back			= 8
    land			= 9
    mailbomb		= 10
    neptune			= 11
    pod				= 12
    processtable	= 13
    smurf			= 14
    teardrop		= 15
    udpstorm		= 16
    buffer_overflow	= 17
    httptunnel		= 18
    loadmodule		= 19
    perl 			= 20
    ps 				= 21
    rootkit			= 22
    sqlattack		= 23
    xterm			= 24
    ftp_write		= 25
    guess_passwd	= 26
    imap			= 27
    multihop		= 28
    named			= 29
    phf				= 30
    sendmail		= 31
    snmpgetattack	= 32
    snmpguess		= 33
    spy				= 34
    warezclient		= 35
    warezmaster		= 36
    worm			= 37
    xlock			= 38
    xsnoop			= 39

    def fromName(name: str):
        for attack in Attack:
            if attack.name == name:
                return attack
        print(name)
    
    def fromValue(value: int):
        for attack in Attack:
            if attack.value == value:
                return attack


class OneHot(object):
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self) -> None:
        self.__protocol_one_hot = np.eye(len(Procotol))
        self.__service_one_hot = np.eye(len(Service))
        self.__flag_one_hot = np.eye(len(Flag))
        self.__attack_one_hot = np.eye(len(Attack))
    
    @property
    def protocol(self):
        return self.__protocol_one_hot
    @property
    def service(self):
        return self.__service_one_hot
    @property
    def flag(self):
        return self.__flag_one_hot
    @property
    def attack(self):
        return self.__attack_one_hot

class DataLoader:
    def __init__(self, src) -> None:
        self.src = src
        self.mid = src + '.mid'
        self.dst = src + '.csv'

    def load(self, cover = False):
        while os.path.exists(self.dst):
            if not cover:
                data = np.genfromtxt(self.dst, delimiter=',')
                return data[:,:-1], data[:, -1]
            os.remove(self.dst)
        self.transform()
        return self.normalizate()

    def transform(self):
        log.info('>>> Start property mapping ...')
        # newline用于解决空行问题
        o_mid = open(self.mid, 'w', newline='')
        with open(self.src, 'r') as o_src:
            csv_reader = csv.reader(o_src)
            csv_writer = csv.writer(o_mid)
            self.rows = 0
            for row in csv_reader:
                # 类型转换
                line = DataLoader._transform(row)
                csv_writer.writerow(line)
                self.rows += 1
            o_mid.close()
        log.info('>>> End property mapping ...')

    def normalizate(self):
        log.info('>>> Start normalizating ...')
        data = np.genfromtxt(self.mid, delimiter=',')
        features, labels = data[:,:-1], data[:, -1]
        dmax, dmin = np.max(features, axis=1), np.min(features, axis=1)
        dmax, dmin = dmax.reshape((-1, 1)), dmin.reshape((-1, 1))
        features = (features - dmin) / (dmax - dmin)
        data = np.c_[features, labels]
        # 不使用numpy.savetxt，数据膨胀严重(200M ---> 1.5G)
        o_dst = open(self.dst, 'w', newline='')
        csv_writer = csv.writer(o_dst)
        csv_writer.writerows(data.tolist())
        log.info('>>> End normalizating ...')
        return (features, labels)

    @staticmethod
    def _transform(line):
        # 字符型 ===> 枚举值
        protocol = Procotol.fromName(line[1])
        ## 测试集有两条icmp的非法数据，需要删掉
        service = Service.fromName(line[2])
        flag = Flag.fromName(line[3])
        # attack = Attack.fromName(line[-1][:-1])
        assial = Assail.toAssail(line[-1][:-1])
        # 枚举值 ===> 独热码
        onehot = OneHot()
        nline = []              # 这里丢弃line[0]
        nline.extend(onehot.protocol[protocol.value - 1])
        nline.extend(onehot.service[service.value - 1])
        nline.extend(onehot.flag[flag.value])
        nline.extend(line[4:-1])
        # 交叉熵会自动将输入标签转换为独热码，这里不需要处理
        nline.append(assial.value)
        return nline
        