---
title: "Simiki + Github Pages 在Mac下搭建部署个人wiki"
layout: page
date: 2018-06-27 00:00
---

[TOC]
## Simiki介绍
[Simiki](http://simiki.org/) 是一个简单的个人Wiki框架。使用Markdown书写Wiki, 生成静态HTML页面。Wiki源文件按目录分类存放, 方便管理维护。

### 目录结构

- `_config.yml`	站点配置文件. 
	 `fabfile.py`	扩展脚本, 提供一些方便的额外操作. 需要安装Fabric.
	 `content`	存储源文件(目前暂时只支持Markdown)的目录.源文件以子目录名分类存放.
如content/linux/bash.md表示bash.md这个源文件属于linux分类.
注意: Simiki 暂时只支持二级目录分类
	 `output`	输出的静态文件(html)目录.
注意: 此目录的生成/更新过程中会存在删除的操作, 请不要将无关且重要的文件放在此目录下
	 `themes`	存储所有主题的目录. 一个主题一个子目录, 全部存放在此目录下.
`_config.yml`配置当前使用的主题
使用simiki新建初始化(simiki init)之后会出现四个文件或文件夹，其中content下存储的是源文件(目前只支持markdown)，后面使用simiki g编译的时候会将content下的文件编译成静态文件目录，并存放到output中.

------
##  Mac下部署到Github Pages


### 准备工作

 1. 注册github账号，创建`<username>.github.io`项目 例如：[2kkkkk.github.io](https://github.com/2kkkkk/2kkkkk.github.io)
 2. 创建`<username>.wiki`项目，并创建`gh-pages`分支 例如：[2kkkkk.wiki](https://github.com/2kkkkk/wiki)
 3. 安装simiki `pip install simiki` 

### 本地初始化
在github中新建完wiki仓库后，clone到本地

    git clone https://github.com/yourUsername/wiki.git

（注意：`git clone` 命令默认clone `master`分支的仓库）

进入wiki文件夹，然后simiki初始化

    cd wiki
    simiki init
    simiki q

此时原来clone下来的wiki空文件夹里面会产生前面目录结构里的那些文件和文件夹。
### 部署到github.io
这时候如何将github.io与这个wiki仓库链接起来呢？
simiki官网给了一个方法：安装Fabric，并且在生成的_config.yml中添加deploy配置项。

**安装Fabirc前需要先安装 [ghp-import](https://github.com/davisp/ghp-import)，一个基于 Python 的工具**

    git clone https://github.com/davisp/ghp-import
    cd ghp-import
    python setup.py install

**安装Fabric：**

    pip install fabric3
（注意：由于是python 3 环境，所以用`pip install fabric3`，如果直接 `pip install fabric`会出错）

**在 _config.yml 中添加 deploy 配置项代码：**

    deploy：
        - type: git
          remote: origin
          branch: gh-pages

该段代码的作用就是将output目录下生成的子文件或者子文件目录，基于git的方式，推送到远程仓库相应的`gh-pages`分支下。

**确认本地和远程仓库关联** 执行 `git remote -v` 可以看到远程仓库的 url

**最后执行部署命令`fab delpoy`即可**

-----

## 网页发布流程

目前simiki仅支持markdown的格式，因此我们每次的文章都需要用markdown的形式来书写，同时还要注意每个markdown都需要添加类似于头文件的东西，如

    ---
    title: "Getting Started"
    layout: page
    date: 2099-06-02 00:00
    ---
### 提交md文件到`master`分支
首先，将`master`分支克隆到本地

    git clone  https://github.com/2kkkkk/wiki.git
    cd wiki

命令`git branch`可以查看本地分支
命令 `git config --global http.postBuffer 1048576000` 可以设置缓冲区大小

将写好的md文件放到`content`文件夹下，`git status`查看工作区状态，理论上会出现content新添加的文件，然后执行

    git add . 
    git commit -m "update"
    git push -u origin master
即可将md文件提交到`master`分支。
### 提交html文件到`gh-pages`分支
> 将`gh-pages`分支pull到本地（这一步很重要，否则fab deploy会报错！！）

这一步其实也不需要，之前报错的原因可能是因为网络原因传输错误，导致远程仓库和本地仓库冲突导致，正常情况下不需要，直接执行`fab deploy` 即可

    git pull origin gh-pages:gh-pages 
执行`simiki g`,编译成功后执行`fab delpoy`，即可将`output`文件夹中的html文件推送到`wiki`仓库的gh-pages分支，这时候就可以在`<yourUserName>.github.io/wiki`下看到你发布的内容了。
**注意，这里有坑：有时候，`fab deploy` 会报如下错误：**

    jackdeMacBook-Pro:wiki jack$ fab deploy
    [localhost] local: which ghp-import > /dev/null 2>&1; echo $?
    [localhost] local: ghp-import -p -m "Update output documentation" -r origin -b gh-pages output
    To https://github.com/2kkkkk/wiki.git
     ! [rejected]        gh-pages -> gh-pages (non-fast-forward)
    error: failed to push some refs to 'https://github.com/2kkkkk/wiki.git'
    hint: Updates were rejected because a pushed branch tip is behind its remote
    hint: counterpart. Check out this branch and integrate the remote changes
    hint: (e.g. 'git pull ...') before pushing again.
    hint: See the 'Note about fast-forwards' in 'git push --help' for details.
    Traceback (most recent call last):
      File "/Users/jack/anaconda3/bin/ghp-import", line 11, in <module>
        load_entry_point('ghp-import==0.5.5', 'console_scripts', 'ghp-import')()
      File "/Users/jack/anaconda3/lib/python3.6/site-packages/ghp_import-0.5.5-py3.6.egg/ghp_import.py", line 244, in main
        git.check_call('push', opts.remote, opts.branch)
      File "/Users/jack/anaconda3/lib/python3.6/site-packages/ghp_import-0.5.5-py3.6.egg/ghp_import.py", line 106, in check_call
        sp.check_call(['git'] + list(args), **kwargs)
      File "/Users/jack/anaconda3/lib/python3.6/subprocess.py", line 291, in check_call
        raise CalledProcessError(retcode, cmd)
    subprocess.CalledProcessError: Command '['git', 'push', 'origin', 'gh-pages']' returned non-zero exit status 1.
    
    Fatal error: local() encountered an error (return code 1) while executing 'ghp-import -p -m "Update output documentation" -r origin -b gh-pages output'
    
    Aborting.

**百度后原因应该是github上的版本和本地版本冲突，因此fab deploy 之前先执行`git pull origin gh-pages:gh-pages`即可**


### 附：Mac下使用Git上传本地项目到github
配置 ssh , 输入命令生成ssh key：`ssh-keygen -t rsa -C "你登录github的邮箱"` 

输入命令，将你的ssh代码复制到剪贴板：`pbcopy < ~/.ssh/id_rsa.pub` 

回到Github上，点击头像进入设置，再进入`SSH and GPG keys`，点击 `New SSH key` 

`Title`：填写你注册的邮箱号，这里就是568581045@qq.com 
`key` ：填写你的生成的id_rsa.pub， 直接Crl＋v将刚才你已经复制在剪贴板里的 ssh 复制到 key input 里面

然后点击 `Add SSH key`

测试是否链接成功，输入命令：`ssh -T -v @git@github.com`

当successfully之后，在 `git config` 里设置一下你的 github 登录名以及登陆邮箱，执行以下两个命令：

    git config --global user.name "your name" 
    
    git config --global user.email "your_email@youremail.com"

至此，下面就可以开始上传代码了。

执行命令：`git status`，就可以看到项目的改动
（由于output下存的是静态页面，每次添加都会改变，所以我们可以在gitignore中将其忽略提交：输入 `touch .gitignore`，生成`.gitignore`文件，在`.gitignore` 文件里输入你要忽略的文件夹及其文件就可以了）


然后执行：`git add .`   (这个点表示更改所有的改动)，

然后执行命令：`git commit -m` "update"

然后执行命令：
`git remote add origin https://github.com/你的用户名/github项目名.git`

最后就执行命令：`git push -u origin master`

则大功告成

------

## Gitment 评论模块

主题`simple`存放在`theme`文件夹下，可以配置主题和页面信息等

[Gitment：使用 GitHub Issues 搭建评论系统](https://imsun.net/posts/gitment-introduction/)
[Gitment评论功能接入踩坑教程](https://www.jianshu.com/p/57afa4844aaa)

##补充：
`simiki g`命令需要用到content和theme两个文件夹的内容来生产output文件夹，因此图片应该放在themes/simple/static/images下

## 后续

markdown文件采用Typora软件编写

1. **提交md文件和相关图片到`master`分支**
   - 将新写好的md文件放到`content`文件夹中，相关的图片放到`themes\simple\static\images`对应文件夹下
   - `git status`命令查看仓库状态，`git add .`命令添加文件到暂存区，`git commit -m 'tijiao'`命令提交到本地仓库，`git push -u origin master `命令提交到远程`wiki`仓库`master`分支
2. **提交html文件到`gh-pages`分支**
   - 每次执行simiki g 都会重新生成所有的html文件，因此先将之前的所有html文件备份，然后用`simiki g`命令编译
   - 由于直接编译生成的html公式显示有问题，因此用Typora直接导出新编写的md文件为html文件，合并入之前的备份下的含有html的对应文件夹中，然后替换掉`output`下的对应文件夹
   - 执行`fab deploy`提交`output`文件夹到远程`wiki`仓库`gh-pages`分支
## 2018/08/14更新
费了半天劲才找出`simiki g`生成的页面公式显示不正确的原因，原来是公式渲染用到的是mathjax，而mathjax与markdwon解析器存在语法冲突：mathjax认为下划线\_表示下标，而markdown解析器认为下划线_表示斜体，因此就出现了公式中本来是下标的部分变成了斜体。  
解决方法是修改markdown解析语法或者修改解析顺序，但操作难度太大，能力有限啊。。。还有一种办法就是在下划线等冲突符号前加一个转义字符\\进行转义，用这种方法的话，如果 想要实时预览，就要找到能同时支持markdown和mathjax的博客平台，真的是太难找了，不过还好我没放弃，终于邂逅了FarBox，完美支持markdown和mathjax，跟`simiki g`生成的网页一毛一样。但是问题又来了，Farbox的linux版本官方很早就不再维护了，最新的版本在ubuntu16.04中无法运行，因此只能转向windwos系统。  
最终解决方案：win10+simiki+FarBox
### win10下simiki+ github pages搭建部署个人wiki
####1 安装simiki
首先装好python和pip，然后用命令`pip install simiki`即可安装simiki。注意：如果是用anaconda安装的python，会出现pyyaml包无法卸载的问题，需要用`conda uninstall pyyaml`先将conda管理的yaml包卸载，然后再`pip install simiki`，就可以解决包冲突问题了。最后再`conda install anaconda-navigator`将刚才卸载pyyaml时附加卸载的navigator装上就可以了。 
windows安装git：https://msysgit.github.com/
####2 部署到github pages
#####2.1 配置ssh
检查本机是否有ssh key设置`cd ~/.ssh`，如果没有则提示： No such file or director。如果有则进入~/.ssh路径下（ls查看当前路径文件，rm * 删除所有文件）
`cd ~`
`·ssh-keygen -t rsa -C "你登录github的邮箱"`
登录github，点击头像进入设置，进入`SSH and GPG keys`，点击`New SSH key`，复制id_rsa.pub的公钥内容到key输入框中，title输入框随便输入，最后点击`Add Key`
测试是否链接成功，`ssh -T -v git@github.com`
设置帐户：

    git config --global user.name "2kkkkk" 
    git config --global user.email "568581045@qq.com"
#####2.2 部署到github
如果不是第一次部署，可以直接跳到第3步
1.先将`master`分支克隆到本地：`git clone git@github.com:2kkkkk/wiki.git`
2.为确保用ssh而不是http连接，可以更新下origin：

    git remote remove origin
    git remote add origin git@github.com:2kkkkk/wiki.git
3.将写好的md文件放进`content`文件夹下，然后命令行进入到`wiki`文件夹：`cd wiki`，执行脚本：`deploy heiheihei`（如果是第一次部署的话，需要先执行`deploy -i`,再执行`deploy heiheihei` ）
