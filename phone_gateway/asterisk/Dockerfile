FROM alpine:3.20

RUN apk add --no-cache asterisk ca-certificates

RUN mkdir -p /run/asterisk /var/log/asterisk /var/spool/asterisk /var/lib/asterisk/agi-bin \
    && chown -R asterisk:asterisk /run/asterisk /var/log/asterisk /var/spool/asterisk /var/lib/asterisk /etc/asterisk

COPY asterisk.conf /etc/asterisk/asterisk.conf
COPY modules.conf /etc/asterisk/modules.conf
COPY logger.conf /etc/asterisk/logger.conf
COPY stasis.conf /etc/asterisk/stasis.conf
COPY manager.conf /etc/asterisk/manager.conf
COPY pjsip.conf /etc/asterisk/pjsip.conf
COPY extensions.conf /etc/asterisk/extensions.conf
COPY rtp.conf /etc/asterisk/rtp.conf

EXPOSE 5060/udp 5060/tcp 5038/tcp 10000-10050/udp

CMD ["sh", "-c", "exec asterisk -f -q -U asterisk -G asterisk > /var/log/asterisk/console.log 2>&1"]
